"""
Utility functions for the activation analysis pipeline.
"""
import os
import torch
import numpy as np
import json
import logging
import hashlib
import h5py
from datetime import datetime
import glob
import pandas as pd

# Set up logging
def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Configure logging to output to both console and file with timestamps."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"analysis_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log the start of the session and the log file location
    logging.info(f"Logging session started. Log file: {log_file}")
    
    return log_file

# Set up logging by default
setup_logging()

# Module logger
logger = logging.getLogger(__name__)

def extract_checkpoint_number(ckpt_path: str):
    """Extract the checkpoint number from a checkpoint path."""
    base = os.path.basename(ckpt_path)
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        return num_str

def standardize(X):
    """Standardize a tensor by removing mean and scaling by standard deviation."""
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    X_std = (X - mean) / std
    return X_std, mean, std

def unstandardize_coefficients(beta_std, mean, std):
    """Convert standardized regression coefficients back to original scale."""
    beta0_std = beta_std[0:1, :]
    beta_rest_std = beta_std[1:, :]
    beta_rest = beta_rest_std / std.T
    beta0 = beta0_std - (mean / std).matmul(beta_rest_std)
    beta = torch.cat([beta0, beta_rest], dim=0)
    return beta

def compute_variance_explained(X_std):
    """Compute variance explained ratios from SVD of standardized data."""
    U, S, Vh = torch.linalg.svd(X_std, full_matrices=False)
    explained_variance = S**2
    total_variance = explained_variance.sum()
    explained_ratio = (explained_variance / total_variance).cpu().numpy()
    cumulative = np.cumsum(explained_ratio)
    return cumulative, S.cpu().numpy()

def report_variance_explained(X_std, report_variance=True):
    """Report variance explained by principal components."""
    if report_variance:
        cumulative, singular_values = compute_variance_explained(X_std)
        return str(cumulative.tolist()), singular_values
    return "", None

def get_markov_cache_key(process_config, max_order):
    """Generate a cache key for Markov approximation data."""
    # Sort the dictionary to ensure consistent hashing
    sorted_config = dict(sorted(process_config.items()))
    # Convert to JSON string and encode
    config_str = json.dumps(sorted_config, sort_keys=True)
    # Add max_order to the hash
    config_str += f"_max_order_{max_order}"
    # Create hash
    return hashlib.md5(config_str.encode()).hexdigest()

def save_results_to_h5(base_dir, run_id, combined_df, best_weights_dict, best_singular_dict, name_suffix=""):
    """
    Save results to H5 file format.
    
    Parameters:
    ----------
    base_dir : str
        Base directory for output files
    run_id : str
        Identifier for the run
    combined_df : pandas.DataFrame
        DataFrame containing regression results
    best_weights_dict : dict
        Dictionary containing best weights
    best_singular_dict : dict
        Dictionary containing singular values
    name_suffix : str, optional
        Suffix to add to the filename (default: "")
        Used to distinguish between different types of runs (e.g., random baseline)
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Determine if this is a random baseline run based on name_suffix
    is_random = name_suffix == "_random_baseline"
    
    # Decide filename based on whether we're using unified storage or not
    h5_path = os.path.join(base_dir, f"{run_id}_results{name_suffix}.h5")
    
    with h5py.File(h5_path, 'w') as f:
        # Store DataFrame as a dataset
        df_group = f.create_group("regression_results")
        for col in combined_df.columns:
            data = combined_df[col].values
            # Convert object columns to strings
            if combined_df[col].dtype == 'object':
                data = np.array([str(x) for x in data], dtype='S100')
            df_group.create_dataset(col, data=data)
        
        # Store metadata about the run type
        f.attrs["is_random_baseline"] = is_random
        
        # Store best weights
        weights_group = f.create_group("best_weights")
        for target, target_weights in best_weights_dict.items():
            target_group = weights_group.create_group(target)
            for layer, layer_data in target_weights.items():
                layer_group = target_group.create_group(layer)
                layer_group.create_dataset("weights", data=layer_data["weights"])
                layer_group.attrs["checkpoint"] = layer_data["checkpoint"]
                layer_group.attrs["rcond"] = layer_data["rcond"]
                layer_group.attrs["dist"] = layer_data["dist"]
        
        # Store singular values
        sv_group = f.create_group("singular_values")
        for target, target_sv in best_singular_dict.items():
            target_group = sv_group.create_group(target)
            for layer, layer_sv_list in target_sv.items():
                layer_group = target_group.create_group(layer)
                for idx, sv_data in enumerate(layer_sv_list):
                    entry = layer_group.create_group(f"entry_{idx}")
                    if "checkpoint" in sv_data:
                        entry.attrs["checkpoint"] = sv_data["checkpoint"]
                    if "random_idx" in sv_data:
                        entry.attrs["random_idx"] = sv_data["random_idx"]
                    entry.create_dataset("singular_values", data=sv_data["singular_values"])
    
    logger.info(f"Saved results to {h5_path}")
    return h5_path

def save_unified_results(output_path, checkpoint_data, random_data=None, compress=True):
    """
    Save both checkpoint and random baseline results in a unified H5 format.
    
    Args:
        output_path: Path to save the unified H5 file
        checkpoint_data: Dictionary containing checkpoint results
        random_data: Optional dictionary containing random baseline results
        compress: Whether to use gzip compression for datasets (default: True)
    """
    compression = 'gzip' if compress else None
    
    with h5py.File(output_path, 'w') as f:
        # Add top-level attributes
        attrs = checkpoint_data.get('attrs', {})
        for k, v in attrs.items():
            f.attrs[k] = v
        
        # Create checkpoint_results group
        checkpoint_group = f.create_group('checkpoint_results')
        
        # Save regression results for checkpoints
        df = checkpoint_data.get('df', None)
        if df is not None:
            df_group = checkpoint_group.create_group('regression_results')
            for col in df.columns:
                df_group.create_dataset(col, data=df[col].values, compression=compression)
        
        # Save best weights for checkpoints
        weights = checkpoint_data.get('weights', {})
        if weights:
            weights_group = checkpoint_group.create_group('best_weights')
            for target, target_weights in weights.items():
                target_group = weights_group.create_group(target)
                for layer, layer_data in target_weights.items():
                    layer_group = target_group.create_group(layer)
                    
                    w_array = layer_data.get('weights', None)
                    if w_array is not None:
                        layer_group.create_dataset('weights', data=w_array, compression=compression)
                        
                    # Add attributes
                    for attr in ['rcond', 'dist']:
                        if attr in layer_data:
                            layer_group.attrs[attr] = layer_data[attr]
                    
                    # Add checkpoint attribute if available
                    if 'checkpoint' in layer_data:
                        layer_group.attrs['checkpoint'] = layer_data['checkpoint']
        
        # Save singular values for checkpoints - ENSURE THIS IS WORKING
        singular_values = checkpoint_data.get('singular', {})
        if singular_values:
            singular_group = checkpoint_group.create_group('singular_values')
            for target, target_sv in singular_values.items():
                target_group = singular_group.create_group(target)
                for layer, layer_entries in target_sv.items():
                    layer_group = target_group.create_group(layer)
                    
                    for entry in layer_entries:
                        # Use checkpoint number in the group name instead of generic entry_i
                        checkpoint_id = entry.get('checkpoint', None)
                        if checkpoint_id is not None:
                            entry_name = f"checkpoint_{checkpoint_id}"
                        else:
                            # Fallback to entry_i if no checkpoint information available
                            static_id = id(entry) % 1000  # Use object id for uniqueness
                            entry_name = f"entry_{static_id}"
                            
                        entry_group = layer_group.create_group(entry_name)
                        sv_array = entry.get('singular_values', None)
                        
                        if sv_array is not None:
                            entry_group.create_dataset('singular_values', data=sv_array, compression=compression)
                            
                        # Keep attributes for backward compatibility
                        for attr in ['checkpoint', 'random_idx']:
                            if attr in entry and entry[attr] is not None:
                                entry_group.attrs[attr] = entry[attr]
        
        # Save random baseline results if provided
        if random_data is not None:
            random_group = f.create_group('random_results')
            
            # Save regression results for random baselines
            df = random_data.get('df', None)
            if df is not None:
                df_group = random_group.create_group('regression_results')
                for col in df.columns:
                    df_group.create_dataset(col, data=df[col].values, compression=compression)
            
            # Save singular values for random baselines
            singular_values = random_data.get('singular', {})
            if singular_values:
                singular_group = random_group.create_group('singular_values')
                for target, target_sv in singular_values.items():
                    target_group = singular_group.create_group(target)
                    for layer, layer_entries in target_sv.items():
                        layer_group = target_group.create_group(layer)
                        
                        for entry in layer_entries:
                            # Use random_idx in the group name
                            random_id = entry.get('random_idx', None)
                            if random_id is not None:
                                entry_name = f"random_{random_id}"
                            else:
                                # Fallback to entry_i if no random_idx information available
                                static_id = id(entry) % 1000  # Use object id for uniqueness
                                entry_name = f"entry_{static_id}"
                            
                            entry_group = layer_group.create_group(entry_name)
                            sv_array = entry.get('singular_values', None)
                            
                            if sv_array is not None:
                                entry_group.create_dataset('singular_values', data=sv_array, compression=compression)
                                
                            # Keep random_idx attribute for backward compatibility
                            if 'random_idx' in entry and entry['random_idx'] is not None:
                                entry_group.attrs['random_idx'] = entry['random_idx']
            
            # Save weights for random baselines (NEW)
            weights = random_data.get('weights', {})
            if weights:
                weights_group = random_group.create_group('best_weights')
                for target, target_weights in weights.items():
                    target_group = weights_group.create_group(target)
                    for layer, layer_data in target_weights.items():
                        layer_group = target_group.create_group(layer)
                        
                        w_array = layer_data.get('weights', None)
                        if w_array is not None:
                            layer_group.create_dataset('weights', data=w_array, compression=compression)
                            
                        # Add attributes
                        for attr in ['rcond', 'dist', 'random_idx']:
                            if attr in layer_data:
                                layer_group.attrs[attr] = layer_data[attr]

def save_results_csv(output_dir, run_id, checkpoint_data, random_data=None):
    """
    Save results in the new CSV+NPY format, replacing the H5 format.
    
    Args:
        output_dir: Base directory to save the results
        run_id: Identifier for the run
        checkpoint_data: Dictionary containing checkpoint results 
                        (df, weights, singular, attrs)
        random_data: Optional dictionary containing random baseline results
    
    Returns:
        dict: Paths to saved files
    """
    # Create required directories
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, "weights")
    singular_values_dir = os.path.join(output_dir, "singular_values")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(singular_values_dir, exist_ok=True)
    
    saved_files = {
        'results_csv': None,
        'random_results_csv': None,
        'metadata_json': None,
        'weight_files': [],
        'singular_value_files': []
    }
    
    # Prepare metadata
    metadata = {
        "run_id": run_id,
        "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_index": {
            "results_csv": None,
            "random_results_csv": None,
            "weights": {},
            "singular_values": {}
        }
    }
    
    # Add any additional attributes from checkpoint_data
    attrs = checkpoint_data.get('attrs', {})
    for k, v in attrs.items():
        if k not in metadata:  # Don't overwrite existing metadata
            metadata[k] = v
    
    # Save checkpoint regression results to CSV
    df = checkpoint_data.get('df', None)
    if df is not None:
        results_csv_path = os.path.join(output_dir, f"{run_id}_results.csv")
        df.to_csv(results_csv_path, index=False)
        metadata["file_index"]["results_csv"] = results_csv_path
        saved_files['results_csv'] = results_csv_path
        logging.info(f"Saved regression results to {results_csv_path}")
    
    # Save checkpoint weights as NPY files
    weights = checkpoint_data.get('weights', {})
    if weights:
        for target, target_weights in weights.items():
            if target not in metadata["file_index"]["weights"]:
                metadata["file_index"]["weights"][target] = {}
                
            for layer, layer_data in target_weights.items():
                if layer not in metadata["file_index"]["weights"][target]:
                    metadata["file_index"]["weights"][target][layer] = {}
                
                w_array = layer_data.get('weights', None)
                if w_array is not None:
                    checkpoint = layer_data.get('checkpoint', 'unknown')
                    weight_file = f"{run_id}_{target}_{layer}_{checkpoint}.npy"
                    weight_path = os.path.join(weights_dir, weight_file)
                    
                    np.save(weight_path, w_array)
                    
                    # Store metadata about this weight file
                    metadata["file_index"]["weights"][target][layer][str(checkpoint)] = {
                        "path": weight_path,
                        "shape": w_array.shape
                    }
                    
                    # Add any additional attributes
                    for attr in ['rcond', 'dist']:
                        if attr in layer_data:
                            metadata["file_index"]["weights"][target][layer][str(checkpoint)][attr] = layer_data[attr]
                    
                    saved_files['weight_files'].append(weight_path)
                    logging.info(f"Saved weights to {weight_path}")
    
    # Save checkpoint singular values as NPY files
    singular_values = checkpoint_data.get('singular', {})
    if singular_values:
        for target, target_sv in singular_values.items():
            if target not in metadata["file_index"]["singular_values"]:
                metadata["file_index"]["singular_values"][target] = {}
                
            for layer, layer_entries in target_sv.items():
                if layer not in metadata["file_index"]["singular_values"][target]:
                    metadata["file_index"]["singular_values"][target][layer] = {}
                
                for entry in layer_entries:
                    sv_array = entry.get('singular_values', None)
                    if sv_array is not None:
                        checkpoint = entry.get('checkpoint', 'unknown')
                        sv_file = f"{run_id}_{target}_{layer}_{checkpoint}.npy"
                        sv_path = os.path.join(singular_values_dir, sv_file)
                        
                        np.save(sv_path, sv_array)
                        
                        # Store metadata about this singular value file
                        metadata["file_index"]["singular_values"][target][layer][str(checkpoint)] = {
                            "path": sv_path,
                            "shape": sv_array.shape
                        }
                        
                        saved_files['singular_value_files'].append(sv_path)
                        logging.info(f"Saved singular values to {sv_path}")
    
    # Save random baseline data if provided
    if random_data is not None:
        # Save random regression results to CSV
        df = random_data.get('df', None)
        if df is not None:
            random_csv_path = os.path.join(output_dir, f"{run_id}_random_results.csv")
            df.to_csv(random_csv_path, index=False)
            metadata["file_index"]["random_results_csv"] = random_csv_path
            saved_files['random_results_csv'] = random_csv_path
            logging.info(f"Saved random baseline results to {random_csv_path}")
        
        # Save random weights as NPY files
        weights = random_data.get('weights', {})
        if weights:
            for target, target_weights in weights.items():
                if target not in metadata["file_index"]["weights"]:
                    metadata["file_index"]["weights"][target] = {}
                    
                for layer, layer_data in target_weights.items():
                    if layer not in metadata["file_index"]["weights"][target]:
                        metadata["file_index"]["weights"][target][layer] = {}
                    
                    w_array = layer_data.get('weights', None)
                    if w_array is not None:
                        random_idx = layer_data.get('random_idx', 'unknown')
                        weight_file = f"{run_id}_{target}_{layer}_random_{random_idx}.npy"
                        weight_path = os.path.join(weights_dir, weight_file)
                        
                        np.save(weight_path, w_array)
                        
                        # Store metadata about this weight file
                        random_key = f"random_{random_idx}"
                        metadata["file_index"]["weights"][target][layer][random_key] = {
                            "path": weight_path,
                            "shape": w_array.shape,
                            "is_random": True,
                            "random_idx": random_idx
                        }
                        
                        # Add any additional attributes
                        for attr in ['rcond', 'dist']:
                            if attr in layer_data:
                                metadata["file_index"]["weights"][target][layer][random_key][attr] = layer_data[attr]
                        
                        saved_files['weight_files'].append(weight_path)
                        logging.info(f"Saved random weights to {weight_path}")
        
        # Save random singular values as NPY files
        singular_values = random_data.get('singular', {})
        if singular_values:
            for target, target_sv in singular_values.items():
                if target not in metadata["file_index"]["singular_values"]:
                    metadata["file_index"]["singular_values"][target] = {}
                    
                for layer, layer_entries in target_sv.items():
                    if layer not in metadata["file_index"]["singular_values"][target]:
                        metadata["file_index"]["singular_values"][target][layer] = {}
                    
                    for entry in layer_entries:
                        sv_array = entry.get('singular_values', None)
                        if sv_array is not None:
                            random_idx = entry.get('random_idx', 'unknown')
                            sv_file = f"{run_id}_{target}_{layer}_random_{random_idx}.npy"
                            sv_path = os.path.join(singular_values_dir, sv_file)
                            
                            np.save(sv_path, sv_array)
                            
                            # Store metadata about this singular value file
                            random_key = f"random_{random_idx}"
                            metadata["file_index"]["singular_values"][target][layer][random_key] = {
                                "path": sv_path,
                                "shape": sv_array.shape,
                                "is_random": True,
                                "random_idx": random_idx
                            }
                            
                            saved_files['singular_value_files'].append(sv_path)
                            logging.info(f"Saved random singular values to {sv_path}")
    
    # Save metadata JSON
    metadata_path = os.path.join(output_dir, f"{run_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    saved_files['metadata_json'] = metadata_path
    logging.info(f"Saved metadata to {metadata_path}")
    
    return saved_files

def load_results_csv(run_id, base_dir):
    """
    Load results from the CSV+NPY format.
    
    Args:
        run_id: Identifier for the run
        base_dir: Base directory where results are stored
    
    Returns:
        dict: Loaded data in a format compatible with the old H5 structure
    """
    results = {
        'checkpoint_df': None,
        'random_df': None,
        'weights': {},
        'singular_values': {},
        'attrs': {}
    }
    
    # Load metadata
    metadata_path = os.path.join(base_dir, f"{run_id}_metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Extract attributes
        for k, v in metadata.items():
            if k != "file_index":  # Skip the file index
                results['attrs'][k] = v
                
        # Load regression results CSV
        results_csv = metadata["file_index"].get("results_csv")
        if results_csv and os.path.exists(results_csv):
            results['checkpoint_df'] = pd.read_csv(results_csv)
            # Ensure is_random flag is set
            if 'is_random' not in results['checkpoint_df'].columns:
                results['checkpoint_df']['is_random'] = False
        
        # Load random results CSV if available
        random_csv = metadata["file_index"].get("random_results_csv")
        if random_csv and os.path.exists(random_csv):
            results['random_df'] = pd.read_csv(random_csv)
            # Ensure is_random flag is set
            if 'is_random' not in results['random_df'].columns:
                results['random_df']['is_random'] = True
        
        # Load weights
        for target, target_data in metadata["file_index"].get("weights", {}).items():
            results['weights'][target] = {}
            for layer, layer_data in target_data.items():
                results['weights'][target][layer] = {}
                for checkpoint, file_info in layer_data.items():
                    is_random = file_info.get("is_random", False)
                    weight_path = file_info.get("path")
                    
                    if weight_path and os.path.exists(weight_path):
                        weights = np.load(weight_path)
                        entry_data = {'weights': weights}
                        
                        # Add metadata
                        for key, value in file_info.items():
                            if key not in ['path', 'shape']:
                                entry_data[key] = value
                        
                        # Set checkpoint or random_idx 
                        if is_random and 'random_idx' in file_info:
                            entry_data['random_idx'] = file_info['random_idx']
                        elif not is_random and checkpoint.isdigit():
                            entry_data['checkpoint'] = int(checkpoint)
                        
                        results['weights'][target][layer] = entry_data
        
        # Load singular values
        for target, target_data in metadata["file_index"].get("singular_values", {}).items():
            results['singular_values'][target] = {}
            for layer, layer_data in target_data.items():
                results['singular_values'][target][layer] = []
                
                for checkpoint, file_info in layer_data.items():
                    is_random = file_info.get("is_random", False)
                    sv_path = file_info.get("path")
                    
                    if sv_path and os.path.exists(sv_path):
                        sv_array = np.load(sv_path)
                        entry_data = {'singular_values': sv_array}
                        
                        # Add metadata
                        for key, value in file_info.items():
                            if key not in ['path', 'shape']:
                                entry_data[key] = value
                        
                        # Set checkpoint or random_idx
                        if is_random and 'random_idx' in file_info:
                            entry_data['random_idx'] = file_info['random_idx']
                        elif not is_random and checkpoint.isdigit():
                            entry_data['checkpoint'] = int(checkpoint)
                        
                        results['singular_values'][target][layer].append(entry_data)
                        
    except Exception as e:
        logging.error(f"Error loading CSV+NPY results: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    return results

def find_results_csv_files(directory):
    """
    Find CSV result files in a directory.
    
    Args:
        directory: Directory to search
    
    Returns:
        tuple: (results_csv_files, random_results_csv_files)
    """
    results_csv_files = {}
    random_results_csv_files = {}
    
    # Look for metadata files to identify runs
    pattern = os.path.join(directory, '*_metadata.json')
    for metadata_file in glob.glob(pattern):
        try:
            # Extract run_id from the filename
            filename = os.path.basename(metadata_file)
            run_id = filename.replace('_metadata.json', '')
            
            # Load metadata to get CSV file paths
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            file_index = metadata.get('file_index', {})
            
            # Check if results CSV exists
            results_csv = file_index.get('results_csv')
            if results_csv and os.path.exists(results_csv):
                results_csv_files[run_id] = {
                    'type': 'csv',
                    'metadata': metadata_file,
                    'results': results_csv
                }
            
            # Check if random results CSV exists
            random_csv = file_index.get('random_results_csv')
            if random_csv and os.path.exists(random_csv):
                random_results_csv_files[run_id] = {
                    'type': 'csv',
                    'metadata': metadata_file,
                    'results': random_csv
                }
                
        except Exception as e:
            logging.error(f"Error processing metadata file {metadata_file}: {e}")
    
    return results_csv_files, random_results_csv_files

def save_regression_results(output_dir, run_id, dataframe, is_random=False):
    """
    Save regression results to CSV file.
    
    Args:
        output_dir: Directory to save the results
        run_id: Identifier for the run
        dataframe: DataFrame containing regression results
        is_random: Whether this is a random baseline result
    
    Returns:
        str: Path to the saved CSV file
    """
    # Create run_id specific directory
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Determine filename based on whether it's a random baseline
    if is_random:
        filename = f"{run_id}_random_baseline.csv"
    else:
        filename = f"{run_id}_regression_results.csv"
        
    file_path = os.path.join(run_dir, filename)
    
    # Save DataFrame to CSV
    dataframe.to_csv(file_path, index=False)
    
    logging.info(f"Saved {'random baseline' if is_random else 'regression'} results to {file_path}")
    return file_path

def save_weights_csv(output_dir, run_id, target, weights_dict):
    """
    Save weights for a specific target to a CSV file.
    
    Args:
        output_dir: Directory to save the results
        run_id: Identifier for the run
        target: Target name (e.g., 'nn_beliefs', 'markov_order_1')
        weights_dict: Dictionary of weights structured as {layer: {checkpoint: data}}
    
    Returns:
        str: Path to the saved CSV file
    """
    # Create run_id specific directory and weights subdirectory
    run_dir = os.path.join(output_dir, run_id)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Create filename for this target's weights
    filename = f"{run_id}_{target}_weights.csv"
    file_path = os.path.join(weights_dir, filename)
    
    # Prepare list to collect weight data rows
    weights_data = []
    
    # Process the weights dictionary
    for layer_name, layer_data in weights_dict.items():
        # Check if this is a random layer by looking for "_random_" in the name
        is_random = '_random_' in layer_name
        random_idx = None
        if is_random:
            # Extract random_idx from layer name
            try:
                random_idx = int(layer_name.split('_random_')[-1])
            except (ValueError, IndexError):
                # If we can't extract a valid random_idx, just continue with it as None
                pass
                
        # Normalize the layer name by removing any "_random_X" suffix
        base_layer = layer_name
        if is_random:
            base_layer = '_'.join(layer_name.split('_random_')[:-1])
        
        # If this is a dictionary with checkpoint keys, process each checkpoint
        if isinstance(layer_data, dict) and any(isinstance(k, (int, str)) and isinstance(v, dict) for k, v in layer_data.items()):
            for checkpoint_id, weight_info in layer_data.items():
                # Handle both tensor and dictionary types
                if isinstance(weight_info, dict):
                    # Dictionary case
                    w_array = weight_info.get('weights')
                    rcond = weight_info.get('rcond', None)
                    dist = weight_info.get('dist', None)
                    
                    # Check if random_idx is in the weight_info
                    if random_idx is None and 'random_idx' in weight_info:
                        is_random = True
                        random_idx = weight_info.get('random_idx')
                else:
                    # Tensor case
                    w_array = weight_info
                    rcond = None
                    dist = None
                
                if w_array is not None:
                    # Convert numpy array to JSON-serializable list
                    weights_json = json.dumps(w_array.tolist() if hasattr(w_array, 'tolist') else w_array)
                    
                    # Create row data with all metadata in a single row
                    row = {
                        'layer_name': base_layer,
                        'checkpoint': checkpoint_id,
                        'weights_data': weights_json,
                        'dims': str(w_array.shape if hasattr(w_array, 'shape') else []),
                        'rcond': rcond,
                        'dist': dist,
                        'is_random': is_random
                    }
                    
                    # Add random_idx if this is a random baseline
                    if is_random and random_idx is not None:
                        row['random_idx'] = random_idx
                    
                    weights_data.append(row)
        else:
            # Handle the case where layer_data is not a dictionary of checkpoints
            # This is a direct weights object with metadata
            if isinstance(layer_data, dict):
                w_array = layer_data.get('weights')
                rcond = layer_data.get('rcond', None)
                dist = layer_data.get('dist', None)
                
                # Check if random_idx is in the layer_data
                if random_idx is None and 'random_idx' in layer_data:
                    is_random = True
                    random_idx = layer_data.get('random_idx')
                
                checkpoint_id = layer_data.get('checkpoint', 'unknown')
                
                if w_array is not None:
                    # Convert numpy array to JSON-serializable list
                    weights_json = json.dumps(w_array.tolist() if hasattr(w_array, 'tolist') else w_array)
                    
                    # Create row data with all metadata in a single row
                    row = {
                        'layer_name': base_layer,
                        'checkpoint': checkpoint_id,
                        'weights_data': weights_json,
                        'dims': str(w_array.shape if hasattr(w_array, 'shape') else []),
                        'rcond': rcond,
                        'dist': dist,
                        'is_random': is_random
                    }
                    
                    # Add random_idx if this is a random baseline
                    if is_random and random_idx is not None:
                        row['random_idx'] = random_idx
                    
                    weights_data.append(row)
    
    # Convert to DataFrame and save
    if weights_data:
        df = pd.DataFrame(weights_data)
        # Reorder columns to ensure weights_data (the JSON column) is last
        cols = [col for col in df.columns if col != 'weights_data'] + ['weights_data']
        df = df[cols]
        df.to_csv(file_path, index=False)
        logging.info(f"Saved weights for target '{target}' to {file_path}")
        return file_path
    else:
        logging.warning(f"No weight data found for target '{target}'")
        return None

def save_singular_values_csv(output_dir, run_id, target, singular_values_dict):
    """
    Save singular values for a specific target to a CSV file.
    
    Args:
        output_dir: Directory to save the results
        run_id: Identifier for the run
        target: Target name (e.g., 'nn_beliefs', 'markov_order_1')
        singular_values_dict: Dictionary of singular values structured as {layer: [entries]}
    
    Returns:
        str: Path to the saved CSV file
    """
    # Create run_id specific directory and singular_values subdirectory
    run_dir = os.path.join(output_dir, run_id)
    singular_values_dir = os.path.join(run_dir, "singular_values")
    os.makedirs(singular_values_dir, exist_ok=True)
    
    # Create filename for this target's singular values
    filename = f"{run_id}_{target}_singular_values.csv"
    file_path = os.path.join(singular_values_dir, filename)
    
    # Prepare list to collect singular value data rows
    sv_data = []
    
    # Process the singular values dictionary
    for layer, layer_entries in singular_values_dict.items():
        # Remove any "_random_X" suffix from layer names to normalize them
        base_layer = layer
        if '_random_' in layer:
            base_layer = '_'.join(layer.split('_random_')[:-1])
            
        for entry in layer_entries:
            sv_array = entry.get('singular_values')
            if sv_array is not None:
                # Convert singular values array to JSON string
                if isinstance(sv_array, list):
                    # Handle case where singular_values is already a list
                    sv_json = json.dumps(sv_array)
                else:
                    # Handle case where singular_values is a numpy array
                    sv_json = json.dumps(sv_array.tolist())
                
                # Create row data
                row = {
                    'layer': base_layer,  # Use normalized layer name
                    'singular_data': sv_json,
                    'dims': str(np.shape(sv_array))
                }
                
                # Add checkpoint or random_idx information
                if 'checkpoint' in entry:
                    row['checkpoint'] = entry['checkpoint']
                    row['is_random'] = False
                elif 'random_idx' in entry:
                    row['checkpoint'] = f"random_{entry['random_idx']}"
                    row['is_random'] = True
                    row['random_idx'] = entry['random_idx']
                
                sv_data.append(row)
    
    # Convert to DataFrame and save
    if sv_data:
        df = pd.DataFrame(sv_data)
        # Reorder columns to ensure singular_data (the JSON column) is last
        cols = [col for col in df.columns if col != 'singular_data'] + ['singular_data']
        df = df[cols]
        df.to_csv(file_path, index=False)
        logging.info(f"Saved singular values for target '{target}' to {file_path}")
        return file_path
    else:
        logging.warning(f"No singular value data found for target '{target}'")
        return None

def save_metadata(output_dir, run_id, metadata_dict):
    """
    Save metadata and file index for a run.
    
    Args:
        output_dir: Directory to save the metadata
        run_id: Identifier for the run
        metadata_dict: Dictionary containing metadata and file index
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create run_id specific directory
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create metadata file path
    file_path = os.path.join(run_dir, f"{run_id}_metadata.json")
    
    # Add format version and timestamp if not present
    if 'format_version' not in metadata_dict:
        metadata_dict['format_version'] = 'csv_v1'
    if 'creation_time' not in metadata_dict:
        metadata_dict['creation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save metadata as JSON
    with open(file_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    logging.info(f"Saved metadata to {file_path}")
    return file_path

def save_results_in_csv_format(output_dir, run_id, checkpoint_data, random_data=None):
    """
    Main function to save all results in the new CSV format.
    
    Args:
        output_dir: Base directory to save the results
        run_id: Identifier for the run
        checkpoint_data: Dictionary containing checkpoint results 
                        (df, weights, singular, attrs)
        random_data: Optional dictionary containing random baseline results
    
    Returns:
        dict: Dictionary of saved file paths
    """
    # Create run-specific directory
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    saved_files = {
        'regression_results': None,
        'random_baseline': None,
        'weights_files': {},
        'singular_values_files': {},
        'metadata': None
    }
    
    # Prepare metadata
    metadata = {
        'run_id': run_id,
        'format_version': 'csv_v1',
        'creation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'file_index': {
            'regression_results': None,
            'random_baseline': None,
            'weights': {},
            'singular_values': {}
        }
    }
    
    # Add any custom attributes
    attrs = checkpoint_data.get('attrs', {})
    for k, v in attrs.items():
        if k not in metadata:  # Don't overwrite existing metadata
            metadata[k] = v
    
    # Save checkpoint regression results
    df = checkpoint_data.get('df', None)
    if df is not None:
        csv_path = save_regression_results(output_dir, run_id, df, is_random=False)
        metadata['file_index']['regression_results'] = csv_path
        saved_files['regression_results'] = csv_path
    
    # Save random baseline results
    if random_data is not None and 'df' in random_data:
        random_csv_path = save_regression_results(output_dir, run_id, random_data['df'], is_random=True)
        metadata['file_index']['random_baseline'] = random_csv_path
        saved_files['random_baseline'] = random_csv_path
    
    # Process and save weights
    # First gather all targets from both checkpoint and random data
    all_weight_targets = set()
    if checkpoint_data and 'weights' in checkpoint_data:
        all_weight_targets.update(checkpoint_data['weights'].keys())
    if random_data and 'weights' in random_data:
        all_weight_targets.update(random_data['weights'].keys())
    
    # Now process each target
    for target in all_weight_targets:
        # Prepare a combined dictionary for this target
        combined_weights = {}
        
        # Add checkpoint weights if available
        checkpoint_weights = checkpoint_data.get('weights', {}).get(target, {})
        for layer, layer_data in checkpoint_weights.items():
            combined_weights[layer] = layer_data
        
        # Add random weights if available
        random_weights = random_data.get('weights', {}).get(target, {}) if random_data else {}
        for layer, layer_data in random_weights.items():
            # Use the layer as is - the random_idx is stored in the data
            combined_weights[layer] = layer_data
        
        # Save the combined weights
        if combined_weights:
            weights_csv = save_weights_csv(output_dir, run_id, target, combined_weights)
            if weights_csv:
                metadata['file_index']['weights'][target] = weights_csv
                saved_files['weights_files'][target] = weights_csv
    
    # Process and save singular values
    # First gather all targets from both checkpoint and random data
    all_sv_targets = set()
    if checkpoint_data and 'singular' in checkpoint_data:
        all_sv_targets.update(checkpoint_data['singular'].keys())
    if random_data and 'singular' in random_data:
        all_sv_targets.update(random_data['singular'].keys())
    
    # Now process each target
    for target in all_sv_targets:
        # Prepare a combined dictionary for this target
        combined_sv = {}
        
        # Add checkpoint singular values if available
        checkpoint_sv = checkpoint_data.get('singular', {}).get(target, {})
        for layer, layer_entries in checkpoint_sv.items():
            combined_sv[layer] = list(layer_entries)  # Create a new list
        
        # Add random singular values if available
        random_sv = random_data.get('singular', {}).get(target, {}) if random_data else {}
        for layer, layer_entries in random_sv.items():
            if layer not in combined_sv:
                combined_sv[layer] = list(layer_entries)  # Create a new list
            else:
                # Append random entries to existing layer entries
                combined_sv[layer].extend(layer_entries)
        
        # Save the combined singular values
        if combined_sv:
            sv_csv = save_singular_values_csv(output_dir, run_id, target, combined_sv)
            if sv_csv:
                metadata['file_index']['singular_values'][target] = sv_csv
                saved_files['singular_values_files'][target] = sv_csv
    
    # Save metadata
    metadata_path = save_metadata(output_dir, run_id, metadata)
    saved_files['metadata'] = metadata_path
    
    return saved_files

def load_results_csv_format(run_id, base_dir):
    """
    Load results from the CSV format.
    
    Args:
        run_id: Identifier for the run
        base_dir: Base directory where results are stored
    
    Returns:
        dict: Loaded data in a format compatible with the dashboard
    """
    results = {
        'checkpoint_df': None,
        'random_df': None,
        'weights': {},
        'singular_values': {},
        'attrs': {}
    }
    
    # Load metadata
    run_dir = os.path.join(base_dir, run_id)
    metadata_path = os.path.join(run_dir, f"{run_id}_metadata.json")
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract attributes
        for k, v in metadata.items():
            if k != 'file_index':  # Skip the file index
                results['attrs'][k] = v
        
        # Load regression results
        regression_csv = metadata.get('file_index', {}).get('regression_results')
        if regression_csv and os.path.exists(regression_csv):
            results['checkpoint_df'] = pd.read_csv(regression_csv)
            # Ensure is_random flag is set
            if 'is_random' not in results['checkpoint_df'].columns:
                results['checkpoint_df']['is_random'] = False
        
        # Load random baseline results
        random_csv = metadata.get('file_index', {}).get('random_baseline')
        if random_csv and os.path.exists(random_csv):
            results['random_df'] = pd.read_csv(random_csv)
            # Ensure is_random flag is set
            if 'is_random' not in results['random_df'].columns:
                results['random_df']['is_random'] = True
        
        # Load weights for each target
        for target, weights_csv in metadata.get('file_index', {}).get('weights', {}).items():
            if os.path.exists(weights_csv):
                weights_df = pd.read_csv(weights_csv)
                results['weights'][target] = {}
                
                # Process each row in the weights DataFrame
                for _, row in weights_df.iterrows():
                    layer = row['layer_name']
                    checkpoint = row['checkpoint']
                    is_random = row.get('is_random', False)
                    
                    # Parse the weights data from JSON
                    weights_data = json.loads(row['weights_data'])
                    weights_array = np.array(weights_data)
                    
                    # Create the weight entry
                    weight_entry = {
                        'weights': weights_array,
                        'rcond': row.get('rcond'),
                        'dist': row.get('dist')
                    }
                    
                    # Add checkpoint or random_idx
                    if is_random:
                        random_idx = row.get('random_idx')
                        if random_idx is not None:
                            weight_entry['random_idx'] = random_idx
                    else:
                        weight_entry['checkpoint'] = checkpoint
                    
                    # Add to results
                    if layer not in results['weights'][target]:
                        results['weights'][target][layer] = {}
                    
                    results['weights'][target][layer][checkpoint] = weight_entry
        
        # Load singular values for each target
        for target, sv_csv in metadata.get('file_index', {}).get('singular_values', {}).items():
            if os.path.exists(sv_csv):
                sv_df = pd.read_csv(sv_csv)
                results['singular_values'][target] = {}
                
                # Process each row in the singular values DataFrame
                for _, row in sv_df.iterrows():
                    layer = row['layer']
                    
                    # Parse the singular values data from JSON
                    sv_data = json.loads(row['singular_data'])
                    sv_array = np.array(sv_data)
                    
                    # Create the singular values entry
                    sv_entry = {
                        'singular_values': sv_array
                    }
                    
                    # Add checkpoint or random_idx
                    if row.get('is_random', False):
                        random_idx = row.get('random_idx')
                        if random_idx is not None:
                            sv_entry['random_idx'] = random_idx  # Don't convert to int, use raw value
                    else:
                        sv_entry['checkpoint'] = row['checkpoint']
                    
                    # Add to results
                    if layer not in results['singular_values'][target]:
                        results['singular_values'][target][layer] = []
                    
                    results['singular_values'][target][layer].append(sv_entry)
                    
    except Exception as e:
        logging.error(f"Error loading CSV format results: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    return results

def find_csv_format_files(directory):
    """
    Find CSV format result files in a directory.
    
    Args:
        directory: Directory to search
    
    Returns:
        tuple: (results_files, random_baseline_files, metadata_files)
    """
    results_files = {}
    random_baseline_files = {}
    metadata_files = {}
    
    # First, list all run directories
    run_dirs = []
    for item in os.listdir(directory):
        run_dir = os.path.join(directory, item)
        if os.path.isdir(run_dir):
            run_dirs.append(run_dir)
    
    # Look for metadata files in each run directory
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        metadata_file = os.path.join(run_dir, f"{run_id}_metadata.json")
        
        if os.path.exists(metadata_file):
            try:
                # Load metadata to check format version
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Only process files with the correct format version
                if metadata.get('format_version', '').startswith('csv_v'):
                    metadata_files[run_id] = metadata_file
                    
                    # Get regression results file
                    regression_csv = metadata.get('file_index', {}).get('regression_results')
                    if regression_csv and os.path.exists(regression_csv):
                        results_files[run_id] = {
                            'type': 'csv_format',
                            'metadata': metadata_file,
                            'regression_results': regression_csv
                        }
                    
                    # Get random baseline file
                    random_csv = metadata.get('file_index', {}).get('random_baseline')
                    if random_csv and os.path.exists(random_csv):
                        random_baseline_files[run_id] = {
                            'type': 'csv_format',
                            'metadata': metadata_file,
                            'random_baseline': random_csv
                        }
                        
            except Exception as e:
                logging.error(f"Error processing metadata file {metadata_file}: {e}")
    
    return results_files, random_baseline_files, metadata_files