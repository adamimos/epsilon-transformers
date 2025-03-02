"""
Script to precompute Markov approximations for different process configurations.

This script precomputes and caches Markov approximations for specified process configurations,
allowing them to be reused across multiple runs that share the same process parameters.

Usage:
    python -m scripts.activation_analysis.precompute_markov --config path/to/config.json
    python -m scripts.activation_analysis.precompute_markov --process-config '{"type": "even_odd", "n_states": 8}'

"""
import os
import json
import argparse
import logging
import torch
from tqdm.auto import tqdm
import traceback

from scripts.activation_analysis.config import *
from scripts.activation_analysis.utils import setup_logging, get_markov_cache_key
from scripts.activation_analysis.data_loading import ModelDataManager
from scripts.activation_analysis.belief_states import BeliefStateGenerator

# Set up module logger
logger = logging.getLogger("precompute_markov")

def precompute_for_process_config(process_config, max_order=4, device='cpu', sample_model_config=None):
    """
    Precompute Markov approximations for a single process configuration.
    
    Args:
        process_config (dict): Process configuration parameters
        max_order (int): Maximum Markov order to compute
        device (str): Device to use for computation
        sample_model_config (dict, optional): Model configuration to use for context length
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize components
        model_data_manager = ModelDataManager(device=device)
        belief_generator = BeliefStateGenerator(model_data_manager, device=device)
        
        # Check if process_config is valid
        if not isinstance(process_config, dict):
            logger.error(f"Invalid process configuration: {process_config}")
            return False
            
        # Log the full process_config for debugging
        logger.info(f"Processing configuration: {json.dumps(process_config, indent=2)}")
        
        # Create a more meaningful identifier based on available keys
        config_keys = sorted(process_config.keys())
        if config_keys:
            config_id = "_".join(f"{k}:{process_config[k]}" for k in config_keys[:3])
        else:
            config_id = "unknown"
        
        # Check if the data is in the cache
        cache_key = get_markov_cache_key(process_config, max_order)
        logger.info(f"Cache key: {cache_key}")
        
        # Check if markov data is already cached
        cached_data = model_data_manager.load_markov_data(process_config, max_order)
        if cached_data is not None:
            # Log the process config with the better identifier
            logger.info(f"Markov data already cached for process config: {config_id}")
            
            # Check the belief dimensions for each order
            for order, data in enumerate(cached_data, 1):
                _, beliefs, _, _, _ = data
                belief_dim = beliefs.shape[-1]
                logger.info(f"  Order {order}: belief dim = {belief_dim}")
                if belief_dim >= 64:
                    logger.info(f"  Stopping at order {order} (dim >= 64)")
            return True
        
        # Create a dummy run_config if sample_model_config is not provided
        if sample_model_config is None:
            sample_model_config = {
                'n_ctx': 8,  # Default context length
                'n_embd': 64,
                'n_layer': 4
            }
        
        run_config = {
            'process_config': process_config,
            'model_config': sample_model_config
        }
        
        logger.info(f"Computing Markov approximations for process config: {config_id}")
        
        # Generate classical belief states up to max_order
        belief_states = belief_generator.generate_classical_belief_states(
            run_config, max_order=max_order
        )
        
        # Check if cache was successfully created
        new_cached_data = model_data_manager.load_markov_data(process_config, max_order)
        logger.info(f"Cache created after computation: {new_cached_data is not None}")
        
        if belief_states:
            logger.info(f"Successfully computed and cached Markov data for process config: {config_id}")
            return True
        else:
            logger.warning(f"No belief states generated for process config: {config_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error computing Markov approximations: {e}")
        logger.error(traceback.format_exc())
        return False

def precompute_from_config_file(config_file, max_order=4, device='cpu'):
    """
    Precompute Markov approximations for process configurations specified in a config file.
    
    Args:
        config_file (str): Path to the config file containing process configurations
        max_order (int): Maximum Markov order to compute
        device (str): Device to use for computation
        
    Returns:
        int: Number of successfully processed configurations
    """
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        if 'process_configs' in config_data:
            process_configs = config_data['process_configs']
            # If config_data is a dict with process_configs, it might also have model_config
            sample_model_config = config_data.get('model_config', None)
        else:
            process_configs = [config_data]
            # In this case, config_data is already a process config and wouldn't have model_config
            sample_model_config = None
        
        logger.info(f"Found {len(process_configs)} process configurations to precompute")
        
        success_count = 0
        for idx, process_config in enumerate(process_configs):
            logger.info(f"Processing configuration {idx+1}/{len(process_configs)}")
            if precompute_for_process_config(process_config, max_order, device, sample_model_config):
                success_count += 1
        
        return success_count
        
    except Exception as e:
        logger.error(f"Error processing config file {config_file}: {e}")
        logger.error(traceback.format_exc())
        return 0
        
def precompute_for_sweeps(sweeps=None, max_order=4, device='cpu'):
    """
    Precompute Markov approximations for process configurations in specified sweeps.
    
    Args:
        sweeps (dict, optional): Dictionary of sweep_id to model_type. If None, uses SWEEPS from config.
        max_order (int): Maximum Markov order to compute
        device (str): Device to use for computation
        
    Returns:
        int: Number of successfully processed sweeps
    """
    if sweeps is None:
        sweeps = SWEEPS
    
    logger.info(f"Processing {len(sweeps)} sweeps")
    
    model_data_manager = ModelDataManager(device=device)
    success_count = 0
    
    for sweep_id, model_type in sweeps.items():
        logger.info(f"Processing sweep {sweep_id} ({model_type})")
        
        # Get all runs in the sweep
        runs = model_data_manager.list_runs_in_sweep(sweep_id)
        if not runs:
            logger.warning(f"No runs found for sweep {sweep_id}. Skipping.")
            continue
            
        logger.info(f"Found {len(runs)} runs in sweep {sweep_id}")
        processed_configs = set()  # Track unique process configurations already processed
        
        for run_id in runs:
            logger.info(f"Processing run {run_id}")
            
            try:
                # Load a checkpoint to get configuration
                checkpoints = model_data_manager.list_checkpoints(sweep_id, run_id)
                if not checkpoints:
                    logger.warning(f"No checkpoints found for run {run_id}. Skipping.")
                    continue
                    
                _, run_config = model_data_manager.load_checkpoint(sweep_id, run_id, checkpoints[0])
                
                if 'process_config' not in run_config:
                    logger.warning(f"No process configuration found in run {run_id}. Skipping.")
                    continue
                    
                process_config = run_config['process_config']
                model_config = run_config.get('model_config', None)
                
                # Log the full process_config for debugging
                logger.info(f"Process config contents: {json.dumps(process_config, indent=2)}")
                
                # Create a more meaningful identifier based on available keys
                config_keys = sorted(process_config.keys())
                if config_keys:
                    config_id = "_".join(f"{k}:{process_config[k]}" for k in config_keys[:3])
                else:
                    config_id = "unknown"
                
                # Create a config identifier for deduplication
                config_str = str(process_config)
                
                if config_str in processed_configs:
                    logger.info(f"Process configuration for run {run_id} already processed. Skipping.")
                    continue
                
                # Safely log process configuration
                logger.info(f"Processing unique configuration from run {run_id}: {config_id}")
                
                if precompute_for_process_config(process_config, max_order, device, model_config):
                    processed_configs.add(config_str)
                
            except Exception as e:
                logger.error(f"Error processing run {run_id} in sweep {sweep_id}: {e}")
                logger.error(traceback.format_exc())
        
        if processed_configs:
            logger.info(f"Successfully processed {len(processed_configs)} unique configurations from sweep {sweep_id}")
            success_count += 1
    
    logger.info(f"Successfully processed {success_count} sweeps")
    return success_count

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Precompute Markov approximations for process configurations')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to a config file containing process configurations')
    parser.add_argument('--process-config', type=str, default=None,
                      help='JSON string of a process configuration')
    parser.add_argument('--process-type', type=str, default=None,
                      help='Type of process (e.g., "even_odd")')
    parser.add_argument('--n-states', type=int, default=None,
                      help='Number of states for the process')
    parser.add_argument('--max-order', type=int, default=4,
                      help='Maximum Markov order to compute')
    parser.add_argument('--device', type=str, default=None,
                      help=f'Device to use (default: {DEFAULT_DEVICE})')
    parser.add_argument('--all-sweeps', action='store_true',
                      help='Precompute for all sweeps defined in config')
    parser.add_argument('--sweep-id', type=str, default=None,
                      help='Precompute for a specific sweep ID')
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger = logging.getLogger("precompute_markov")
    logger.info(f"Starting Markov approximation precomputation. Log file: {log_file}")
    
    # Set device to use
    device = args.device or DEFAULT_DEVICE
    logger.info(f"Using device: {device}")
    
    # Determine what to precompute
    if args.config:
        # Precompute from config file
        logger.info(f"Precomputing from config file: {args.config}")
        success_count = precompute_from_config_file(args.config, args.max_order, device)
        logger.info(f"Successfully processed {success_count} configurations from config file")
        
    elif args.process_config:
        # Precompute for a specific process configuration from JSON string
        try:
            process_config = json.loads(args.process_config)
            logger.info(f"Precomputing for process config: {process_config}")
            success = precompute_for_process_config(process_config, args.max_order, device)
            logger.info(f"Process completed {'successfully' if success else 'with errors'}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON string: {args.process_config}")
            
    elif args.process_type and args.n_states:
        # Precompute for a specific process type and n_states
        process_config = {
            "type": args.process_type,
            "n_states": args.n_states
        }
        logger.info(f"Precomputing for process type: {args.process_type}, n_states: {args.n_states}")
        success = precompute_for_process_config(process_config, args.max_order, device)
        logger.info(f"Process completed {'successfully' if success else 'with errors'}")
        
    elif args.all_sweeps:
        # Precompute for all sweeps
        logger.info("Precomputing for all sweeps")
        success_count = precompute_for_sweeps(None, args.max_order, device)
        logger.info(f"Successfully processed {success_count} sweeps")
        
    elif args.sweep_id:
        # Precompute for a specific sweep
        if args.sweep_id in SWEEPS:
            logger.info(f"Precomputing for sweep: {args.sweep_id}")
            sweeps = {args.sweep_id: SWEEPS[args.sweep_id]}
            success_count = precompute_for_sweeps(sweeps, args.max_order, device)
            logger.info(f"Successfully processed {success_count} sweeps")
        else:
            logger.error(f"Unknown sweep ID: {args.sweep_id}")
            
    else:
        # No valid options provided
        logger.error("No valid precomputation option provided. Use --help to see available options.")
        parser.print_help()

if __name__ == "__main__":
    main() 