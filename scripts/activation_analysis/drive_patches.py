"""
Patches for the activation_analysis module to make it compatible with GoogleDriveModelLoader.

This module contains patched versions of functions from epsilon_transformers.analysis.activation_analysis 
that accept GoogleDriveModelLoader as an alternative to S3ModelLoader.
"""
import os
import json
import numpy as np
import torch
import io
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Union

# Import the GoogleDriveModelLoader
from epsilon_transformers.analysis.drive_loader import GoogleDriveModelLoader

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can handle numpy arrays and other non-standard types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)

def get_process_hash(process_config: dict) -> str:
    """Generate a hash from a process configuration."""
    # Sort the config to ensure consistent ordering
    sorted_config = {key: process_config[key] for key in sorted(process_config.keys())}
    config_str = json.dumps(sorted_config, sort_keys=True)
    # Create hash
    return hashlib.md5(config_str.encode()).hexdigest()

def get_process_data_path(process_config: dict, loader: Union[GoogleDriveModelLoader, Any]) -> str:
    """Generate the data path for cached process data."""
    process_hash = get_process_hash(process_config)
    return f"analysis/process_data/{process_hash}"

def save_process_data(data: dict, process_config: dict, loader: Union[GoogleDriveModelLoader, Any]):
    """Save process data with the appropriate loader."""
    path = get_process_data_path(process_config, loader)
    
    # Create the folder structure if using GoogleDriveModelLoader
    if isinstance(loader, GoogleDriveModelLoader):
        folder_path = os.path.join(loader.base_drive_path, "analysis", "process_data")
        os.makedirs(folder_path, exist_ok=True)
    
    metadata_key = f"{path}/metadata.json"
    metadata = {
        "process_config": process_config,
        "process_hash": get_process_hash(process_config),
        "creation_time": time.time(),
    }
    
    # Save metadata
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, cls=NumpyEncoder)
    )
    
    # Save each tensor in the data as a separate NPZ file
    for key, tensor in data.items():
        tensor_key = f"{path}/{key}.npz"
        
        # Save tensor as numpy array
        buf = io.BytesIO()
        np.savez_compressed(buf, tensor=tensor.detach().cpu().numpy())
        buf.seek(0)
        
        # Upload to loader
        loader.s3_client.put_object(
            Bucket=loader.bucket_name,
            Key=tensor_key,
            Body=buf.getvalue()
        )
    
    print(f"Saved process data at {path}")

def load_process_data(process_config: dict, loader: Union[GoogleDriveModelLoader, Any]) -> Optional[dict]:
    """Load process data using the appropriate loader."""
    path = get_process_data_path(process_config, loader)
    metadata_key = f"{path}/metadata.json"
    
    try:
        # Check if metadata exists
        response = loader.s3_client.get_object(
            Bucket=loader.bucket_name,
            Key=metadata_key
        )
        
        metadata_str = response['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_str)
        
        # Verify the process hash matches
        if metadata.get('process_hash') != get_process_hash(process_config):
            print("Warning: Process hash mismatch. The cached data may be for a different configuration.")
            return None
        
        # Load all tensor files
        data = {}
        
        # List all files in the path
        if isinstance(loader, GoogleDriveModelLoader):
            process_path = os.path.join(loader.base_drive_path, path)
            if not os.path.exists(process_path):
                return None
            files = [f for f in os.listdir(process_path) if f.endswith('.npz')]
            
            for filename in files:
                key = filename.replace('.npz', '')
                if key != 'metadata':
                    tensor_path = os.path.join(process_path, filename)
                    npz_data = np.load(tensor_path, allow_pickle=True)
                    data[key] = torch.tensor(npz_data['tensor'])
        else:
            # For S3ModelLoader, use the existing method to list objects
            response = loader.s3_client.list_objects_v2(
                Bucket=loader.bucket_name,
                Prefix=f"{path}/"
            )
            
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith('.npz'):
                    tensor_name = key.split('/')[-1].replace('.npz', '')
                    if tensor_name != 'metadata':
                        tensor_response = loader.s3_client.get_object(
                            Bucket=loader.bucket_name,
                            Key=key
                        )
                        
                        with io.BytesIO(tensor_response['Body'].read()) as buf:
                            npz_data = np.load(buf, allow_pickle=True)
                            data[tensor_name] = torch.tensor(npz_data['tensor'])
        
        print(f"Loaded process data from {path}")
        return data
            
    except Exception as e:
        print(f"Error loading process data: {e}")
        return None

def prepare_msp_data(config, model_config, loader: Optional[Union[GoogleDriveModelLoader, Any]] = None):
    """Prepare MSP data with caching for either S3 or Google Drive."""
    if loader is not None:
        # Try to load cached data
        cached_data = load_process_data(config['process_config'], loader)
        if cached_data is not None:
            print("Loading cached MSP data...")
            return (
                cached_data['nn_inputs'],
                cached_data['nn_beliefs'],
                cached_data['nn_indices'],
                cached_data['nn_probs'],
                cached_data.get('nn_unnormalized', None)
            )
        else:
            print("No cached MSP data found. Generating...")

    # Import necessary functions only when needed to avoid circular imports
    from epsilon_transformers.analysis.activation_analysis import get_msp, get_beliefs_for_nn_inputs

    # Get the MSP data
    msp = get_msp(config)
    tree_paths = msp.paths
    tree_beliefs = msp.beliefs
    tree_unnormalized_beliefs = getattr(msp, 'unnormalized_beliefs', None)
    path_length = tree_paths[0][-1] + 1 if tree_paths else 0
    
    # Get unique beliefs
    unique_beliefs = sorted(set(tuple(b) for b in tree_beliefs))
    msp_belief_index = {b: i for i, b in enumerate(unique_beliefs)}
    
    # Get path probs
    path_probs = [msp.get_seq_prob(list(path)) for path in tree_paths]
    probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}
    
    # We need to fix the get_beliefs_for_nn_inputs import
    nn_inputs = torch.tensor(tree_paths)
    
    # Import the original function to use for now
    from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs
    
    # Get beliefs for neural network inputs
    nn_return_vals = get_beliefs_for_nn_inputs(
        nn_inputs,
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized_beliefs,
        probs_dict
    )
    
    # Extract the return values
    if len(nn_return_vals) == 2:
        nn_beliefs, nn_indices = nn_return_vals
        nn_probs = None
        nn_unnormalized = None
    elif len(nn_return_vals) == 3:
        if tree_unnormalized_beliefs is None:
            nn_beliefs, nn_indices, nn_probs = nn_return_vals
            nn_unnormalized = None
        else:
            nn_beliefs, nn_indices, nn_unnormalized = nn_return_vals
            nn_probs = None
    else:  # len(nn_return_vals) == 4
        nn_beliefs, nn_indices, nn_probs, nn_unnormalized = nn_return_vals
    
    # Cache the data if a loader is provided
    if loader is not None:
        cache_data = {
            'nn_inputs': nn_inputs,
            'nn_beliefs': nn_beliefs,
            'nn_indices': nn_indices,
            'nn_probs': nn_probs if nn_probs is not None else torch.zeros_like(nn_indices, dtype=torch.float32),
        }
        if nn_unnormalized is not None:
            cache_data['nn_unnormalized'] = nn_unnormalized
        
        # Save to cache
        save_process_data(cache_data, config['process_config'], loader)
    
    return nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized

def save_analysis_results(loader: Union[GoogleDriveModelLoader, Any], sweep_id: str, run_id: str, 
                         checkpoint_key: str, results: dict, title: str = ""):
    """Save analysis results using the appropriate loader."""
    start_time = time.time()
    
    # Check for loader type
    if isinstance(loader, GoogleDriveModelLoader):
        base_path = os.path.join(loader.base_drive_path, "analysis", sweep_id, run_id)
        os.makedirs(base_path, exist_ok=True)
    
    # Extract checkpoint number
    checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
    
    # Save overall metadata
    metadata_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/{title}_metadata.json"
    metadata = {
        'sweep_id': sweep_id,
        'run_id': run_id,
        'checkpoint': checkpoint_num,
        'creation_time': time.time(),
        'mse': results.get('mse')
    }
    
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=metadata_key,
        Body=json.dumps(metadata, cls=NumpyEncoder)
    )
    
    # Save layer results
    for layer_dict in results['layers']:
        layer_name = layer_dict['layer_name']
        
        # Ensure the directory exists if using GoogleDriveModelLoader
        if isinstance(loader, GoogleDriveModelLoader):
            layer_path = os.path.join(base_path, f"checkpoint_{checkpoint_num}", "layers", layer_name)
            os.makedirs(layer_path, exist_ok=True)
        
        # Save regression data
        regression_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/layers/{layer_name}/{title}_regression.json"
        regression_data = {
            'layer_name': layer_name,
            'mse': layer_dict['mse'],
            'mse_shuffled': layer_dict['mse_shuffled'],
            'mse_cv': layer_dict['mse_cv'],
            'regression_coef': layer_dict['regression_coef'].tolist() if isinstance(layer_dict['regression_coef'], (np.ndarray, torch.Tensor)) else layer_dict['regression_coef'],
            'regression_intercept': layer_dict['regression_intercept'].tolist() if isinstance(layer_dict['regression_intercept'], (np.ndarray, torch.Tensor)) else layer_dict['regression_intercept']
        }
        
        loader.s3_client.put_object(
            Bucket=loader.bucket_name,
            Key=regression_key,
            Body=json.dumps(regression_data, cls=NumpyEncoder)
        )

        # Save large arrays
        for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
            array_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/layers/{layer_name}/{title}_{array_name}.npy"
            buf = io.BytesIO()
            np.save(buf, layer_dict[array_name])
            buf.seek(0)
            loader.s3_client.put_object(
                Bucket=loader.bucket_name,
                Key=array_key,
                Body=buf.getvalue()
            )
    
    # Save all_layers results similarly
    if 'all_layers' in results:
        all_layers = results['all_layers']
        
        # Ensure the directory exists if using GoogleDriveModelLoader
        if isinstance(loader, GoogleDriveModelLoader):
            all_layers_path = os.path.join(base_path, f"checkpoint_{checkpoint_num}", "all_layers")
            os.makedirs(all_layers_path, exist_ok=True)
        
        # Save regression data
        regression_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/all_layers/{title}_regression.json"
        regression_data = {
            'mse': all_layers['mse'],
            'mse_shuffled': all_layers['mse_shuffled'],
            'mse_cv': all_layers['mse_cv'],
            'regression_coef': all_layers['regression_coef'].tolist() if isinstance(all_layers['regression_coef'], (np.ndarray, torch.Tensor)) else all_layers['regression_coef'],
            'regression_intercept': all_layers['regression_intercept'].tolist() if isinstance(all_layers['regression_intercept'], (np.ndarray, torch.Tensor)) else all_layers['regression_intercept']
        }
        
        loader.s3_client.put_object(
            Bucket=loader.bucket_name,
            Key=regression_key,
            Body=json.dumps(regression_data, cls=NumpyEncoder)
        )

        # Save large arrays
        for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
            array_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/all_layers/{title}_{array_name}.npy"
            buf = io.BytesIO()
            np.save(buf, all_layers[array_name])
            buf.seek(0)
            loader.s3_client.put_object(
                Bucket=loader.bucket_name,
                Key=array_key,
                Body=buf.getvalue()
            )

    print(f"Total save time: {time.time() - start_time:.2f}s")

def load_analysis_results(loader: Union[GoogleDriveModelLoader, Any], sweep_id: str, run_id: str, 
                         checkpoint_key: str):
    """Load analysis results using the appropriate loader."""
    checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
    base_prefix = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/"
    
    # Determine available analysis files
    if isinstance(loader, GoogleDriveModelLoader):
        base_path = os.path.join(loader.base_drive_path, "analysis", sweep_id, run_id, f"checkpoint_{checkpoint_num}")
        if not os.path.exists(base_path):
            return None
        
        # Find metadata files to identify available analyses
        metadata_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith('_metadata.json'):
                    metadata_files.append(os.path.join(root, file))
        
        if not metadata_files:
            print(f"No analysis results found for {sweep_id}/{run_id} checkpoint {checkpoint_num}")
            return None
    else:
        try:
            response = loader.s3_client.list_objects_v2(
                Bucket=loader.bucket_name,
                Prefix=base_prefix
            )
            
            if 'Contents' not in response:
                print(f"No analysis results found for {sweep_id}/{run_id} checkpoint {checkpoint_num}")
                return None
                
            metadata_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('_metadata.json')]
            
            if not metadata_files:
                print(f"No analysis results found for {sweep_id}/{run_id} checkpoint {checkpoint_num}")
                return None
        except Exception as e:
            print(f"Error listing analysis results: {e}")
            return None
    
    # Load the first available analysis
    metadata_key = metadata_files[0]
    title = metadata_key.split('/')[-1].replace('_metadata.json', '')
    
    print(f"Loading analysis results for {title}")
    
    # Load metadata
    try:
        if isinstance(loader, GoogleDriveModelLoader):
            with open(metadata_key, 'r') as f:
                metadata = json.load(f)
        else:
            response = loader.s3_client.get_object(
                Bucket=loader.bucket_name,
                Key=metadata_key
            )
            metadata = json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None
    
    # Initialize results dictionary
    results = {
        'metadata': metadata,
        'layers': [],
        'all_layers': None
    }
    
    # Load layer results
    if isinstance(loader, GoogleDriveModelLoader):
        layers_path = os.path.join(base_path, "layers")
        if os.path.exists(layers_path):
            layers = os.listdir(layers_path)
            
            for layer in layers:
                layer_path = os.path.join(layers_path, layer)
                regression_path = os.path.join(layer_path, f"{title}_regression.json")
                
                if os.path.exists(regression_path):
                    with open(regression_path, 'r') as f:
                        layer_data = json.load(f)
                    
                    # Load predictions
                    predictions = {}
                    for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
                        array_path = os.path.join(layer_path, f"{title}_{array_name}.npy")
                        if os.path.exists(array_path):
                            predictions[array_name] = np.load(array_path)
                    
                    # Combine into layer dict
                    layer_dict = {**layer_data, **predictions}
                    results['layers'].append(layer_dict)
    else:
        # For S3ModelLoader, use listing to find layer directories
        try:
            layers_prefix = f"{base_prefix}layers/"
            response = loader.s3_client.list_objects_v2(
                Bucket=loader.bucket_name,
                Prefix=layers_prefix,
                Delimiter='/'
            )
            
            layers = [prefix['Prefix'].split('/')[-2] for prefix in response.get('CommonPrefixes', [])]
            
            for layer in layers:
                layer_prefix = f"{layers_prefix}{layer}/"
                
                # Load regression data
                regression_key = f"{layer_prefix}{title}_regression.json"
                try:
                    regression_response = loader.s3_client.get_object(
                        Bucket=loader.bucket_name,
                        Key=regression_key
                    )
                    layer_data = json.loads(regression_response['Body'].read().decode('utf-8'))
                    
                    # Load predictions
                    predictions = {}
                    for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
                        array_key = f"{layer_prefix}{title}_{array_name}.npy"
                        try:
                            array_response = loader.s3_client.get_object(
                                Bucket=loader.bucket_name,
                                Key=array_key
                            )
                            with io.BytesIO(array_response['Body'].read()) as buf:
                                predictions[array_name] = np.load(buf)
                        except:
                            print(f"Warning: Could not load {array_name} for layer {layer}")
                    
                    # Combine into layer dict
                    layer_dict = {**layer_data, **predictions}
                    results['layers'].append(layer_dict)
                except:
                    print(f"Warning: Could not load regression data for layer {layer}")
        except Exception as e:
            print(f"Error loading layer results: {e}")
    
    # Load all_layers results if available
    all_layers_prefix = f"{base_prefix}all_layers/"
    
    if isinstance(loader, GoogleDriveModelLoader):
        all_layers_path = os.path.join(base_path, "all_layers")
        if os.path.exists(all_layers_path):
            regression_path = os.path.join(all_layers_path, f"{title}_regression.json")
            
            if os.path.exists(regression_path):
                with open(regression_path, 'r') as f:
                    all_layers_data = json.load(f)
                
                # Load predictions
                predictions = {}
                for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
                    array_path = os.path.join(all_layers_path, f"{title}_{array_name}.npy")
                    if os.path.exists(array_path):
                        predictions[array_name] = np.load(array_path)
                
                # Combine into all_layers dict
                results['all_layers'] = {**all_layers_data, **predictions}
    else:
        try:
            # Check if all_layers directory exists
            response = loader.s3_client.list_objects_v2(
                Bucket=loader.bucket_name,
                Prefix=all_layers_prefix,
                MaxKeys=1
            )
            
            if 'Contents' in response:
                # Load regression data
                regression_key = f"{all_layers_prefix}{title}_regression.json"
                try:
                    regression_response = loader.s3_client.get_object(
                        Bucket=loader.bucket_name,
                        Key=regression_key
                    )
                    all_layers_data = json.loads(regression_response['Body'].read().decode('utf-8'))
                    
                    # Load predictions
                    predictions = {}
                    for array_name in ['predictions', 'predictions_shuffled', 'predictions_cv']:
                        array_key = f"{all_layers_prefix}{title}_{array_name}.npy"
                        try:
                            array_response = loader.s3_client.get_object(
                                Bucket=loader.bucket_name,
                                Key=array_key
                            )
                            with io.BytesIO(array_response['Body'].read()) as buf:
                                predictions[array_name] = np.load(buf)
                        except:
                            print(f"Warning: Could not load {array_name} for all_layers")
                    
                    # Combine into all_layers dict
                    results['all_layers'] = {**all_layers_data, **predictions}
                except:
                    print("Warning: Could not load regression data for all_layers")
        except Exception as e:
            print(f"Error checking for all_layers: {e}")
    
    return results 