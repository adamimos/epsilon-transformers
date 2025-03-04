import os
import yaml
import json
import pandas as pd
import torch
import numpy as np
import io
from io import StringIO, BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any
from transformer_lens import HookedTransformer, HookedTransformerConfig
from epsilon_transformers.training.networks import create_RNN

class GoogleDriveModelLoader:
    """Similar to S3ModelLoader but uses Google Drive instead of S3."""
    
    def __init__(self, base_drive_path='/content/drive/My Drive/quantum/'):
        """
        Initialize the Google Drive Model Loader.
        
        Args:
            base_drive_path: The base path in Google Drive where models and data are stored
        """
        self.base_drive_path = base_drive_path
        
    def _check_path_exists(self, path):
        """Check if a path exists in Google Drive."""
        return os.path.exists(path)
    
    def list_sweeps(self):
        """List all sweep directories in the drive path."""
        sweeps = []
        base_path = self.base_drive_path
        
        # List all directories in the base path
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                full_path = os.path.join(base_path, item)
                if os.path.isdir(full_path):
                    sweeps.append(item)
        
        return sorted(sweeps)
    
    def get_sweep_ind(self, sweep_id):
        """Get the index of a sweep in the list of sweeps."""
        return self.list_sweeps().index(sweep_id)
    
    def load_sweep_config(self, sweep_id):
        """Load the sweep configuration YAML file for a given sweep ID."""
        config_path = os.path.join(self.base_drive_path, sweep_id, "sweep_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No sweep_config.yaml found for sweep {sweep_id}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def list_sweep_files(self, sweep_id):
        """List all files directly within a sweep directory."""
        sweep_path = os.path.join(self.base_drive_path, sweep_id)
        files = []
        
        if os.path.exists(sweep_path):
            for item in os.listdir(sweep_path):
                full_path = os.path.join(sweep_path, item)
                if os.path.isfile(full_path):
                    files.append(os.path.join(sweep_id, item))
        
        return sorted(files)
    
    def list_runs_in_sweep(self, sweep_id):
        """List all run directories in a sweep."""
        sweep_path = os.path.join(self.base_drive_path, sweep_id)
        runs = []
        
        if os.path.exists(sweep_path):
            for item in os.listdir(sweep_path):
                full_path = os.path.join(sweep_path, item)
                if os.path.isdir(full_path):
                    runs.append(item)
        
        return sorted(runs)
    
    def list_checkpoints(self, sweep_id, run_id):
        """List all checkpoint files for a specific run."""
        run_path = os.path.join(self.base_drive_path, sweep_id, run_id)
        checkpoints = []
        
        if os.path.exists(run_path):
            for item in os.listdir(run_path):
                if item.endswith('.pt'):
                    checkpoints.append(os.path.join(sweep_id, run_id, item))
        
        return sorted(checkpoints)
    
    def load_loss_from_run(self, sweep_id: str, run_id: str) -> Optional[pd.DataFrame]:
        """Load loss data for a specific run within a sweep."""
        try:
            configs = self.load_run_configs(sweep_id, run_id)
            return configs['loss_csv']
        except Exception as e:
            print(f"Error loading loss data: {e}")
            return None
    
    def load_transformer_checkpoint(self, sweep_id: str, run_id: str, checkpoint_key: str, device: str = 'cpu'):
        """Load a specific transformer checkpoint from Google Drive."""
        # Determine the actual path from checkpoint_key
        checkpoint_path = os.path.join(self.base_drive_path, checkpoint_key)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        
        return model, configs['run_config']
    
    def load_rnn_checkpoint(self, sweep_id, run_id, checkpoint_key, device='cpu'):
        """Load a specific RNN checkpoint from Google Drive."""
        # Determine the actual path from checkpoint_key
        checkpoint_path = os.path.join(self.base_drive_path, checkpoint_key)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load run config
        config_path = os.path.join(self.base_drive_path, sweep_id, run_id, "run_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Run config not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Infer vocab size from the output layer in the state dict
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        output_layer_weight = state_dict['output_layer.weight']
        vocab_size = output_layer_weight.size(0)  # First dimension is output size (vocab size)
        
        # Create RNN model based on config structure
        model_config = config['model_config']
        model = create_RNN(config, vocab_size, device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        return model, config
    
    def check_if_process_data_exists(self, process_folder_name):
        """Check if a process data folder exists in Google Drive."""
        process_path = os.path.join(self.base_drive_path, "analysis", process_folder_name)
        return os.path.exists(process_path)
    
    def load_checkpoint(self, sweep_id, run_id, checkpoint_key, device='cpu'):
        """Load a checkpoint, detecting whether it's an RNN or Transformer model by trying both."""
        try:
            # Try loading as RNN first
            return self.load_rnn_checkpoint(sweep_id, run_id, checkpoint_key, device)
        except Exception as e:
            # If RNN fails, try loading as transformer
            try:
                return self.load_transformer_checkpoint(sweep_id, run_id, checkpoint_key, device)
            except Exception as e2:
                raise ValueError(f"Could not load checkpoint as either RNN or Transformer: {e}, {e2}")
    
    def load_run_configs(self, sweep_id, run_id):
        """Load all configuration files for a specific run."""
        configs = {}
        base_path = os.path.join(self.base_drive_path, sweep_id, run_id)
        
        # Load run_config.yaml
        run_config_path = os.path.join(base_path, "run_config.yaml")
        try:
            if os.path.exists(run_config_path):
                with open(run_config_path, 'r') as f:
                    configs['run_config'] = yaml.safe_load(f)
            else:
                configs['run_config'] = None
        except Exception as e:
            print(f"Error loading run_config.yaml: {e}")
            configs['run_config'] = None
        
        # Try loading hooked_model_config.json first, then model_config.json as fallback
        try:
            hooked_config_path = os.path.join(base_path, "hooked_model_config.json")
            model_config_path = os.path.join(base_path, "model_config.json")
            
            if os.path.exists(hooked_config_path):
                with open(hooked_config_path, 'r') as f:
                    configs['model_config'] = json.load(f)
            elif os.path.exists(model_config_path):
                with open(model_config_path, 'r') as f:
                    configs['model_config'] = json.load(f)
            else:
                configs['model_config'] = None
        except Exception as e:
            print(f"Error loading model config files: {e}")
            configs['model_config'] = None
        
        # Load log.json
        log_json_path = os.path.join(base_path, "log.json")
        try:
            if os.path.exists(log_json_path):
                with open(log_json_path, 'r') as f:
                    configs['log'] = json.load(f)
            else:
                configs['log'] = None
        except Exception as e:
            print(f"Error loading log.json: {e}")
            configs['log'] = None
        
        # Load CSV files as pandas DataFrames
        log_csv_path = os.path.join(base_path, "log.csv")
        try:
            if os.path.exists(log_csv_path):
                configs['log_csv'] = pd.read_csv(log_csv_path)
            else:
                configs['log_csv'] = None
        except Exception as e:
            print(f"Error loading log.csv: {e}")
            configs['log_csv'] = None
        
        loss_csv_path = os.path.join(base_path, "loss.csv")
        try:
            if os.path.exists(loss_csv_path):
                configs['loss_csv'] = pd.read_csv(loss_csv_path)
            else:
                configs['loss_csv'] = None
        except Exception as e:
            print(f"Error loading loss.csv: {e}")
            configs['loss_csv'] = None
        
        return configs
    
    def save_mse_csv(self, sweep_id: str, run_id: str, df: pd.DataFrame) -> None:
        """Save MSE DataFrame to Google Drive as a CSV file."""
        # Create the analysis directory if it doesn't exist
        analysis_dir = os.path.join(self.base_drive_path, "analysis", sweep_id, run_id)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Define the path for the CSV file
        csv_path = os.path.join(analysis_dir, "mse_data.csv")
        
        # Save the DataFrame
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved MSE data to {csv_path}")
    
    def load_mse_csv(self, sweep_id: str, run_id: str) -> pd.DataFrame:
        """Load MSE DataFrame from Google Drive."""
        csv_path = os.path.join(self.base_drive_path, "analysis", sweep_id, run_id, "mse_data.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No MSE data found for sweep {sweep_id}, run {run_id}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Successfully loaded MSE data from {csv_path}")
            return df
        except Exception as e:
            raise Exception(f"Error loading MSE data: {str(e)}")
    
    def check_mse_exists(self, sweep_id: str, run_id: str) -> bool:
        """Check if MSE data exists for a given sweep and run."""
        csv_path = os.path.join(self.base_drive_path, "analysis", sweep_id, run_id, "mse_data.csv")
        return os.path.exists(csv_path)
    
    # Methods for saving and loading analysis results
    def save_object(self, key: str, data: Any) -> None:
        """Save an object to Google Drive (replacement for S3 put_object)."""
        # Extract the directory path
        dir_path = os.path.dirname(os.path.join(self.base_drive_path, key))
        os.makedirs(dir_path, exist_ok=True)
        
        full_path = os.path.join(self.base_drive_path, key)
        
        # Determine the type of data and save accordingly
        if isinstance(data, str):
            with open(full_path, 'w') as f:
                f.write(data)
        elif isinstance(data, bytes):
            with open(full_path, 'wb') as f:
                f.write(data)
        else:
            raise ValueError(f"Unsupported data type for save_object: {type(data)}")
        
        print(f"Successfully saved object to {full_path}")
    
    def get_object(self, key: str) -> Any:
        """Get an object from Google Drive (replacement for S3 get_object)."""
        full_path = os.path.join(self.base_drive_path, key)
        
        if not os.path.exists(full_path):
            class NoSuchKey(Exception):
                pass
            self.s3_client = type('', (), {})
            self.s3_client.exceptions = type('', (), {})
            self.s3_client.exceptions.NoSuchKey = NoSuchKey
            raise self.s3_client.exceptions.NoSuchKey(f"No object found at {full_path}")
        
        # Read file contents
        with open(full_path, 'rb') as f:
            data = f.read()
        
        # Create a response-like object
        return {'Body': BytesIO(data)}
    
    @property
    def bucket_name(self):
        """Legacy property to maintain compatibility with S3ModelLoader."""
        return "google-drive"
    
    @property
    def s3_client(self):
        """Mock s3_client for compatibility with existing code."""
        class MockS3Client:
            def __init__(self, loader):
                self.loader = loader
                self.exceptions = type('', (), {})
                self.exceptions.NoSuchKey = FileNotFoundError
                self.exceptions.ClientError = Exception
            
            def put_object(self, Bucket, Key, Body):
                self.loader.save_object(Key, Body)
            
            def get_object(self, Bucket, Key):
                return self.loader.get_object(Key)
            
            def head_object(self, Bucket, Key):
                full_path = os.path.join(self.loader.base_drive_path, Key)
                if not os.path.exists(full_path):
                    raise Exception("File not found")
                return {}
            
            def list_objects_v2(self, Bucket, Prefix, Delimiter=None):
                prefix_path = os.path.join(self.loader.base_drive_path, Prefix)
                dir_path = os.path.dirname(prefix_path)
                
                result = {'CommonPrefixes': [], 'Contents': []}
                
                if os.path.exists(dir_path):
                    for item in os.listdir(dir_path):
                        full_path = os.path.join(dir_path, item)
                        rel_path = os.path.join(os.path.dirname(Prefix), item)
                        
                        if os.path.isdir(full_path):
                            result['CommonPrefixes'].append({'Prefix': f"{rel_path}/"})
                        else:
                            result['Contents'].append({'Key': rel_path})
                
                return result
        
        return MockS3Client(self) 