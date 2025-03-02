#!/usr/bin/env python
"""
Verify if the Markov approximation cache is working correctly.
This script checks if cached Markov data exists for each sweep and run.
"""

import os
import logging
import json
import argparse
import traceback
from datetime import datetime

from scripts.activation_analysis.config import SWEEPS
from scripts.activation_analysis.utils import setup_logging, get_markov_cache_key
from scripts.activation_analysis.data_loading import ModelDataManager
from scripts.activation_analysis.belief_states import BeliefStateGenerator

# Setup logging
logger = logging.getLogger("verify_cache")
setup_logging()  # Don't pass the logger to this function

def check_local_cache(process_config, max_order=4):
    """Check if the process configuration exists in the local cache."""
    cache_key = get_markov_cache_key(process_config, max_order)
    local_cache_dir = os.path.join("analysis", "local_cache", "markov_data", cache_key)
    
    # Check if the directory exists
    if not os.path.exists(local_cache_dir):
        return False, None
    
    # Look for any order files that exist (don't require all of them)
    existing_files = []
    for order in range(1, max_order + 1):
        file_path = os.path.join(local_cache_dir, f"order_{order}.npz")
        if os.path.exists(file_path):
            existing_files.append(file_path)
    
    # Consider cache valid if at least one order file exists
    return len(existing_files) > 0, existing_files

def verify_cache_for_sweep(sweep_id, model_type, device='cpu'):
    """
    Check if the Markov approximation cache exists for all runs in a sweep.
    
    Args:
        sweep_id (str): Sweep ID to check
        model_type (str): Model type (e.g., 'transformer', 'rnn')
        device (str): Device to use
        
    Returns:
        dict: Dictionary mapping run_id to cache status
    """
    # Initialize with both local and S3 options to check both
    model_data_manager_local = ModelDataManager(device=device, use_local_cache=True)
    model_data_manager_s3 = ModelDataManager(device=device, use_local_cache=False)
    
    results = {}
    
    # Get all runs in the sweep
    runs = model_data_manager_local.list_runs_in_sweep(sweep_id)
    if not runs:
        logger.warning(f"No runs found for sweep {sweep_id}. Skipping.")
        return results
        
    logger.info(f"Found {len(runs)} runs in sweep {sweep_id}")
    
    for run_id in runs:
        logger.info(f"Checking cache for run {run_id}")
        
        try:
            # Load a checkpoint to get configuration
            checkpoints = model_data_manager_local.list_checkpoints(sweep_id, run_id)
            if not checkpoints:
                logger.warning(f"No checkpoints found for run {run_id}. Skipping.")
                results[run_id] = {"status": "no_checkpoints"}
                continue
                
            _, run_config = model_data_manager_local.load_checkpoint(sweep_id, run_id, checkpoints[0])
            
            if 'process_config' not in run_config:
                logger.warning(f"No process configuration found in run {run_id}. Skipping.")
                results[run_id] = {"status": "no_process_config"}
                continue
                
            process_config = run_config['process_config']
            
            # Log the full process_config for debugging
            logger.info(f"Process config: {json.dumps(process_config, indent=2)}")
            
            # Create a more meaningful identifier based on available keys
            config_keys = sorted(process_config.keys())
            if config_keys:
                config_id = "_".join(f"{k}:{process_config[k]}" for k in config_keys[:3])
            else:
                config_id = "unknown"
            
            # Get the cache key
            max_order = 4  # Default max order
            cache_key = get_markov_cache_key(process_config, max_order)
            
            # Check local cache first
            local_cache_exists, local_files = check_local_cache(process_config, max_order)
            
            # Check S3 cache
            s3_cached_data = model_data_manager_s3.load_markov_data(process_config, max_order)
            s3_cache_exists = s3_cached_data is not None
            
            if local_cache_exists:
                # Data exists in local cache
                logger.info(f"✅ Markov data found in LOCAL cache for {config_id}")
                cached_data = model_data_manager_local.load_markov_data(process_config, max_order)
                
                # Check dimensions for each order if data was loaded successfully
                dimensions = []
                if cached_data is not None:
                    for order, data in enumerate(cached_data, 1):
                        _, beliefs, _, _, _ = data
                        belief_dim = beliefs.shape[-1]
                        dimensions.append(belief_dim)
                        logger.info(f"  Order {order}: belief dim = {belief_dim}")
                else:
                    logger.warning(f"⚠️ Failed to load local cache data for {config_id} despite files existing")
                    
                results[run_id] = {
                    "status": "cached" if cached_data is not None else "partial_cache",
                    "cache_source": "local",
                    "config_id": config_id,
                    "cache_key": cache_key,
                    "dimensions": dimensions,
                    "local_files": local_files
                }
            elif s3_cache_exists:
                # Data exists in S3 cache
                logger.info(f"✅ Markov data found in S3 cache for {config_id}")
                
                # Check dimensions for each order
                dimensions = []
                for order, data in enumerate(s3_cached_data, 1):
                    _, beliefs, _, _, _ = data
                    belief_dim = beliefs.shape[-1]
                    dimensions.append(belief_dim)
                    logger.info(f"  Order {order}: belief dim = {belief_dim}")
                    
                results[run_id] = {
                    "status": "cached",
                    "cache_source": "s3",
                    "config_id": config_id,
                    "cache_key": cache_key,
                    "dimensions": dimensions
                }
            else:
                logger.info(f"❌ No cached Markov data for {config_id}")
                results[run_id] = {
                    "status": "not_cached",
                    "config_id": config_id,
                    "cache_key": cache_key
                }
            
        except Exception as e:
            logger.error(f"Error checking cache for run {run_id}: {e}")
            logger.error(traceback.format_exc())
            results[run_id] = {"status": "error", "error": str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Verify Markov approximation cache')
    parser.add_argument('--sweep-id', type=str, help='Specific sweep to check')
    parser.add_argument('--all-sweeps', action='store_true', help='Check all sweeps')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    args = parser.parse_args()
    device = args.device
    
    all_results = {}
    
    if args.sweep_id:
        if args.sweep_id in SWEEPS:
            model_type = SWEEPS[args.sweep_id]
            logger.info(f"Checking cache for sweep: {args.sweep_id} ({model_type})")
            results = verify_cache_for_sweep(args.sweep_id, model_type, device)
            all_results[args.sweep_id] = results
        else:
            logger.error(f"Unknown sweep ID: {args.sweep_id}")
            
    elif args.all_sweeps:
        logger.info("Checking cache for all sweeps")
        for sweep_id, model_type in SWEEPS.items():
            logger.info(f"Checking sweep {sweep_id} ({model_type})")
            results = verify_cache_for_sweep(sweep_id, model_type, device)
            all_results[sweep_id] = results
            
    else:
        logger.error("Please specify --sweep-id or --all-sweeps")
        parser.print_help()
        return
        
    # Generate summary
    cached_count = 0
    local_cached_count = 0
    s3_cached_count = 0
    not_cached_count = 0
    error_count = 0
    
    for sweep_id, sweep_results in all_results.items():
        for run_id, result in sweep_results.items():
            if result.get("status") == "cached":
                cached_count += 1
                if result.get("cache_source") == "local":
                    local_cached_count += 1
                elif result.get("cache_source") == "s3":
                    s3_cached_count += 1
            elif result.get("status") == "not_cached":
                not_cached_count += 1
            else:
                error_count += 1
                
    logger.info("=== Cache Verification Summary ===")
    logger.info(f"Total runs checked: {cached_count + not_cached_count + error_count}")
    logger.info(f"Cached: {cached_count}")
    logger.info(f"  - Local cache: {local_cached_count}")
    logger.info(f"  - S3 cache: {s3_cached_count}")
    logger.info(f"Not cached: {not_cached_count}")
    logger.info(f"Errors: {error_count}")
    
    # Save results to file if requested
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"cache_verification_{timestamp}.json"
        
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 