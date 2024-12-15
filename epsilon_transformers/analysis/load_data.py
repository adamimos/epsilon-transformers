import boto3
import os
import yaml
import json
import pandas as pd
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv
import torch
from epsilon_transformers.training.networks import create_RNN
from transformer_lens import HookedTransformer, HookedTransformerConfig
from typing import Optional

class S3ModelLoader:
    def __init__(self):
        load_dotenv()
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        self.bucket_name = "quantum-runs"


    def list_sweeps(self):
        """List all sweep directories in the bucket"""
        paginator = self.s3_client.get_paginator('list_objects_v2')
        sweeps = set()
        
        for page in paginator.paginate(Bucket=self.bucket_name, Delimiter='/'):
            for prefix_dict in page.get('CommonPrefixes', []):
                sweep_name = prefix_dict.get('Prefix', '').rstrip('/')
                sweeps.add(sweep_name)
                
        return sorted(list(sweeps))
    
    def get_sweep_ind(self, sweep_id):
        """Get the index of a sweep in the list of sweeps"""
        return self.list_sweeps().index(sweep_id)

    def load_sweep_config(self, sweep_id):
        """Load the sweep configuration YAML file for a given sweep ID.
        
        Args:
            sweep_id (str): ID of the sweep to load config for
            
        Returns:
            dict: The loaded sweep configuration
            
        Raises:
            FileNotFoundError: If sweep_config.yaml doesn't exist for this sweep
        """
        key = f"{sweep_id}/sweep_config.yaml"
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            yaml_content = response['Body'].read().decode('utf-8')
            return yaml.safe_load(yaml_content)
        except self.s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"No sweep_config.yaml found for sweep {sweep_id}")


    def list_sweep_files(self, sweep_id):
        """List all files (not directories) directly within a sweep directory"""
        prefix = f"{sweep_id}/"
        paginator = self.s3_client.get_paginator('list_objects_v2')
        files = set()
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Only include files directly in the sweep directory
                # (those that don't have additional '/' after the sweep prefix)
                if key.count('/') == prefix.count('/'):
                    files.add(key)
                
        return sorted(list(files))
    
    def list_runs_in_sweep(self, sweep_id):
        """List all run directories within a sweep"""
        prefix = f"{sweep_id}/"
        paginator = self.s3_client.get_paginator('list_objects_v2')
        runs = set()
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter='/'):
            for prefix_dict in page.get('CommonPrefixes', []):
                run_path = prefix_dict.get('Prefix', '')
                run_name = run_path.rstrip('/').split('/')[-1]
                runs.add(run_name)
                
        return sorted(list(runs))

    def list_checkpoints(self, sweep_id, run_id):
        """List all checkpoint files for a specific run within a sweep"""
        prefix = f"{sweep_id}/{run_id}/"
        checkpoints = []
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.pt'):
                    checkpoints.append(obj['Key'])
        
        # Sort checkpoints numerically based on the number before .pt
        return sorted(checkpoints, 
                     key=lambda x: int(x.split('/')[-1].replace('.pt', '')))
    
    def list_config_files(self, sweep_id, run_id):
        """List all non-checkpoint files in a run directory"""
        prefix = f"{sweep_id}/{run_id}/"
        files = []
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                if not obj['Key'].endswith('.pt'):
                    files.append(obj['Key'])
                    
        return sorted(files)
    

    def load_loss_from_run(self, sweep_id: str, run_id: str) -> Optional[pd.DataFrame]:
        """Load loss data for a specific run within a sweep.
        
        Args:
            sweep_id (str): ID of the sweep
            run_id (str): ID of the run
            
        Returns:
            Optional[pd.DataFrame]: DataFrame containing loss data, or None if not found
        """
        try:
            configs = self.load_run_configs(sweep_id, run_id)
            return configs['loss_csv']
        except Exception as e:
            print(f"Error loading loss data: {e}")
            return None
        

    def load_transformer_checkpoint(self, sweep_id: str, run_id: str, checkpoint_key: str, device: str = 'cpu'):
        """Load a specific transformer checkpoint from S3.
        
        Args:
            sweep_id (str): ID of the sweep
            run_id (str): ID of the run
            checkpoint_idx (int): Index of checkpoint to load (-1 for latest)
            device (str): Device to load model onto ('cpu' or 'cuda')
            
        Returns:
            Tuple[HookedTransformer, dict]: The loaded model and its run configuration
        """
        # Create a temporary directory for downloading
        temp_dir = Path(f"./temp/{sweep_id}/{run_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of checkpoints and select the requested one
        checkpoints = self.list_checkpoints(sweep_id, run_id)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for run {run_id}")
                
        # Download checkpoint file
        checkpoint_path = temp_dir / "model.pt"
        self.s3_client.download_file(
            self.bucket_name,
            checkpoint_key,
            str(checkpoint_path)
        )
        
        # Load configurations
        configs = self.load_run_configs(sweep_id, run_id)
        if not configs['model_config']:
            raise ValueError("Could not load model configuration")
        
        # Prepare model config
        model_config = configs['model_config']
        model_config['dtype'] = getattr(torch, model_config['dtype'].split('.')[-1])
        model_config['device'] = device
        
        # Create and load model
        model_config = HookedTransformerConfig(**model_config)
        model = HookedTransformer(model_config)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        return model, configs['run_config']
    
    def load_rnn_checkpoint(self, sweep_id, run_id, checkpoint_key, device='cpu'):
        """Load a specific RNN checkpoint from S3"""
        # Create a temporary directory for downloading
        temp_dir = Path(f"./temp/{sweep_id}/{run_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Download checkpoint file
        checkpoint_path = temp_dir / "model.pt"
        self.s3_client.download_file(
            self.bucket_name,
            checkpoint_key,
            str(checkpoint_path)
        )

        # Download and load run config
        config_key = f"{sweep_id}/{run_id}/run_config.yaml"
        config_path = temp_dir / "run_config.yaml"
        self.s3_client.download_file(
            self.bucket_name,
            config_key,
            str(config_path)
        )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Infer vocab size from the output layer in the state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        output_layer_weight = state_dict['output_layer.weight']
        vocab_size = output_layer_weight.size(0)  # First dimension is output size (vocab size)

        # Create RNN model based on config structure
        model_config = config['model_config']

        model = create_RNN(config, vocab_size, device)

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        
        return model, config
    
    def check_if_process_data_exists(self,process_folder_name):
        """Check if a process data folder exists in S3"""
        key = f"analysis/{process_folder_name}/"
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=key)
        return 'Contents' in response

    def load_checkpoint(self, sweep_id, run_id, checkpoint_key, device='cpu'):
        """Load a checkpoint, detecting whether it's an RNN or Transformer model by trying both"""
        try:
            # Try loading as RNN first
            return self.load_rnn_checkpoint(sweep_id, run_id, checkpoint_key, device)
        except:
            # If RNN fails, try loading as transformer
            return self.load_transformer_checkpoint(sweep_id, run_id, checkpoint_key, device)
    
    def load_run_configs(self, sweep_id, run_id):
        """Load all configuration files for a specific run"""
        configs = {}
        
        # Helper function to download and read file content
        def read_s3_file(key):
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read().decode('utf-8')

        base_path = f"{sweep_id}/{run_id}"
        
        # Load run_config.yaml
        try:
            yaml_content = read_s3_file(f"{base_path}/run_config.yaml")
            configs['run_config'] = yaml.safe_load(yaml_content)
        except Exception as e:
            print(f"Error loading run_config.yaml: {e}")
            configs['run_config'] = None

        # Try loading hooked_model_config.json first, then model_config.json as fallback
        try:
            try:
                json_content = read_s3_file(f"{base_path}/hooked_model_config.json")
                configs['model_config'] = json.loads(json_content)
            except:
                json_content = read_s3_file(f"{base_path}/model_config.json")
                configs['model_config'] = json.loads(json_content)
        except Exception as e:
            print(f"Error loading model config files: {e}")
            configs['model_config'] = None

        # Load log.json
        try:
            json_content = read_s3_file(f"{base_path}/log.json")
            configs['log'] = json.loads(json_content)
        except Exception as e:
            print(f"Error loading log.json: {e}")
            configs['log'] = None

        # Load CSV files as pandas DataFrames
        try:
            csv_content = read_s3_file(f"{base_path}/log.csv")
            configs['log_csv'] = pd.read_csv(StringIO(csv_content))
        except Exception as e:
            print(f"Error loading log.csv: {e}")
            configs['log_csv'] = None

        try:
            csv_content = read_s3_file(f"{base_path}/loss.csv")
            configs['loss_csv'] = pd.read_csv(StringIO(csv_content))
        except Exception as e:
            print(f"Error loading loss.csv: {e}")
            configs['loss_csv'] = None

        return configs
    