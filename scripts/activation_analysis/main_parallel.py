"""
Parallel execution entry point for the activation analysis pipeline.
This script allows processing specific run_ids on specific GPUs or distributing multiple runs across GPUs.
"""
import os
import torch
import argparse
import logging
import traceback
import json
import subprocess
from datetime import datetime
from collections import defaultdict

from scripts.activation_analysis.config import (
    OUTPUT_DIR, SWEEPS, RCOND_SWEEP_LIST, 
    MAX_MARKOV_ORDER, TRANSFORMER_ACTIVATION_KEYS,
    MAX_CHECKPOINTS, PROCESS_ALL_CHECKPOINTS
)
from scripts.activation_analysis.main import process_run
from scripts.activation_analysis.utils import setup_logging, is_s3_path

# Set up module logger
logger = logging.getLogger("main_parallel")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run activation analysis in parallel')
    
    # Run selection options
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument('--run-id', type=str, default=None,
                      help='Specific run ID to process')
    run_group.add_argument('--auto-detect', action='store_true',
                      help='Auto-detect available runs to process')
    run_group.add_argument('--run-list', type=str, default=None,
                      help='JSON file with list of run IDs to process')
    
    # Distribution options
    parser.add_argument('--distribute', action='store_true',
                      help='Distribute runs across multiple GPUs')
    parser.add_argument('--num-gpus', type=int, default=1,
                      help='Number of GPUs to distribute jobs across (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=0,
                      help='Specific GPU ID to use when processing a single run (default: 0)')
    
    # Processing options
    parser.add_argument('--sweep-id', type=str, default=None,
                      help='Sweep ID to process (default: auto-detect from run ID)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--s3-output', type=str, default=None,
                      help='S3 output path (e.g., s3://bucket-name/path/to/output)')
    parser.add_argument('--log-dir', type=str, default=None,
                      help='Log directory (defaults to OUTPUT_DIR/logs)')
    parser.add_argument('--max-checkpoints', type=int, default=None,
                      help=f'Maximum number of checkpoints to process (default: {MAX_CHECKPOINTS})')
    parser.add_argument('--process-all-checkpoints', action='store_true',
                      help='Process all checkpoints, not just the most recent ones')
    
    # Execution control
    parser.add_argument('--dry-run', action='store_true',
                      help='Print commands without executing them')
    parser.add_argument('--wait', action='store_true',
                      help='Wait for each batch to complete before starting the next')
    
    return parser.parse_args()

def get_run_list(args):
    """Get list of runs to process."""
    run_list = []
    
    # Case 1: Specific run ID provided
    if args.run_id:
        return [args.run_id]
    
    # Case 2: Run list file provided
    if args.run_list and os.path.exists(args.run_list):
        with open(args.run_list, 'r') as f:
            run_list = json.load(f)
        logger.info(f"Loaded {len(run_list)} runs from {args.run_list}")
        return run_list
    
    # Case 3: Auto-detect runs
    if args.auto_detect:
        try:
            from scripts.activation_analysis.data_loading import ModelDataManager
            mdm = ModelDataManager()
            
            for sweep_id, model_type in SWEEPS.items():
                # Skip if a specific sweep was requested and this isn't it
                if args.sweep_id and sweep_id != args.sweep_id:
                    continue
                    
                logger.info(f"Finding runs for sweep {sweep_id}...")
                
                # Get all runs in the sweep with 4 layers (adjust filter as needed)
                runs = [run_id for run_id in mdm.list_runs_in_sweep(sweep_id) 
                        if 'L4' in run_id]
                
                run_list.extend(runs)
                logger.info(f"Found {len(runs)} runs in sweep {sweep_id}")
        except Exception as e:
            logger.error(f"Error auto-detecting runs: {e}")
            logger.error(traceback.format_exc())
            return []
    
    return run_list

def setup_device(gpu_id):
    """Set up device for computation."""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        logger.info(f"Using GPU device: {device}")
    elif torch.backends.mps.is_available():  # For Apple Silicon
        device = "mps"
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
        logger.info("Using CPU device")
    return device

def get_output_dir(args):
    """Get the appropriate output directory from the arguments."""
    # S3 output path takes precedence if specified
    if args.s3_output:
        if not args.s3_output.startswith('s3://'):
            logger.warning(f"S3 output path should start with 's3://'. Got: {args.s3_output}")
            logger.warning(f"Prepending 's3://' to the path")
            return f"s3://{args.s3_output}"
        return args.s3_output
    
    # Otherwise use local output directory
    return args.output_dir or OUTPUT_DIR

def process_single_run(args, sweep_id, run_id, model_type, gpu_id=0):
    """Process a single run on the specified GPU."""
    try:
        device = setup_device(gpu_id)
        logger.info(f"Processing run {run_id} from sweep {sweep_id} ({model_type}) on device {device}")
        
        # Override global variables if needed
        global MAX_CHECKPOINTS, PROCESS_ALL_CHECKPOINTS
        if args.max_checkpoints is not None:
            MAX_CHECKPOINTS = args.max_checkpoints
        if args.process_all_checkpoints:
            PROCESS_ALL_CHECKPOINTS = True
        
        # Get output directory (local or S3)
        output_dir = get_output_dir(args)
        logger.info(f"Using output directory: {output_dir}")
            
        result = process_run(sweep_id, run_id, model_type, device=device, output_dir=output_dir)
        logger.info(result)
        return True
    except Exception as e:
        logger.error(f"Error processing run {run_id} in sweep {sweep_id}:")
        logger.error(traceback.format_exc())
        return False

def distribute_runs(run_list, args):
    """Distribute runs across multiple GPUs."""
    if not run_list:
        logger.warning("No runs to process")
        return
    
    # Distribute runs evenly across GPUs
    gpu_runs = defaultdict(list)
    for i, run_id in enumerate(run_list):
        gpu_id = i % args.num_gpus
        gpu_runs[gpu_id].append(run_id)
    
    # Track processes for each GPU
    processes = []
    success_count = 0
    failure_count = 0
    
    # Process each GPU's runs
    for gpu_id, runs in gpu_runs.items():
        for run_id in runs:
            # Determine sweep_id if needed
            sweep_id = args.sweep_id
            if sweep_id is None:
                for sid in SWEEPS:
                    if sid in run_id:
                        sweep_id = sid
                        break
                
                # If still None, use the first sweep as fallback
                if sweep_id is None and SWEEPS:
                    sweep_id = list(SWEEPS.keys())[0]
                    logger.warning(f"Could not determine sweep_id from run_id. Using first available: {sweep_id}")
            
            if sweep_id not in SWEEPS:
                logger.error(f"Invalid sweep_id: {sweep_id}. Available sweeps: {list(SWEEPS.keys())}")
                continue
            
            model_type = SWEEPS[sweep_id]
            
            # Print the command
            cmd_str = f"Processing run {run_id} on GPU {gpu_id}"
            logger.info(cmd_str)
            
            # Execute or dry-run
            if not args.dry_run:
                # Run directly
                if args.wait:
                    # Process sequentially
                    success = process_single_run(args, sweep_id, run_id, model_type, gpu_id)
                    if success:
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    # Process in parallel using subprocess
                    cmd = ["python", "-m", "scripts.activation_analysis.main_parallel"]
                    cmd.extend(["--run-id", run_id])
                    cmd.extend(["--gpu-id", str(gpu_id)])
                    
                    if sweep_id:
                        cmd.extend(["--sweep-id", sweep_id])
                    
                    # Pass output directory
                    output_dir = get_output_dir(args)
                    if is_s3_path(output_dir):
                        cmd.extend(["--s3-output", output_dir])
                    elif args.output_dir:
                        cmd.extend(["--output-dir", args.output_dir])
                    
                    if args.max_checkpoints is not None:
                        cmd.extend(["--max-checkpoints", str(args.max_checkpoints)])
                    
                    if args.process_all_checkpoints:
                        cmd.append("--process-all-checkpoints")
                    
                    proc = subprocess.Popen(cmd)
                    processes.append((gpu_id, run_id, proc))
            else:
                logger.info(f"[DRY RUN] Would process run {run_id} on GPU {gpu_id}")
    
    # Wait for processes to complete if running in parallel
    if not args.wait and processes and not args.dry_run:
        logger.info("Waiting for all processes to complete...")
        for gpu_id, run_id, proc in processes:
            logger.info(f"Waiting for run {run_id} on GPU {gpu_id}...")
            proc.wait()
            returncode = proc.returncode
            if returncode == 0:
                logger.info(f"Run {run_id} on GPU {gpu_id} completed successfully")
                success_count += 1
            else:
                logger.error(f"Run {run_id} on GPU {gpu_id} failed with return code {returncode}")
                failure_count += 1
    
    if not args.dry_run:
        logger.info(f"Processing complete. Successful runs: {success_count}, Failed runs: {failure_count}")

def main():
    """Main entry point."""
    # Parse arguments
    parser = parse_arguments()
    args = parser
    
    # Get appropriate output directory (local or S3)
    output_dir = get_output_dir(args)
    
    # Create local output directory if needed
    if not is_s3_path(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = args.log_dir or os.path.join(output_dir if not is_s3_path(output_dir) else OUTPUT_DIR, "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(log_dir=log_dir)
    logger.info(f"Starting activation analysis with parallel processing. Log file: {log_file}")
    logger.info(f"Using output directory: {output_dir}")
    
    # Override config settings if specified
    if args.max_checkpoints is not None:
        global MAX_CHECKPOINTS
        MAX_CHECKPOINTS = args.max_checkpoints
        logger.info(f"Using custom MAX_CHECKPOINTS: {MAX_CHECKPOINTS}")
    
    if args.process_all_checkpoints:
        global PROCESS_ALL_CHECKPOINTS
        PROCESS_ALL_CHECKPOINTS = True
        logger.info("Processing all checkpoints")
    
    # Determine execution mode
    if args.distribute or args.auto_detect or args.run_list:
        # Multiple run mode
        run_list = get_run_list(args)
        logger.info(f"Found {len(run_list)} runs to process")
        
        if run_list:
            # Distribute runs across GPUs
            distribute_runs(run_list, args)
        else:
            logger.error("No runs found to process")
    elif args.run_id:
        # Single run mode
        # Determine sweep_id if not provided
        sweep_id = args.sweep_id
        if sweep_id is None:
            # Auto-detect sweep from run_id
            for sid in SWEEPS:
                if sid in args.run_id:
                    sweep_id = sid
                    break
            
            # If still None, use the first sweep as fallback
            if sweep_id is None and SWEEPS:
                sweep_id = list(SWEEPS.keys())[0]
                logger.warning(f"Could not determine sweep_id from run_id. Using first available: {sweep_id}")
        
        if sweep_id not in SWEEPS:
            logger.error(f"Invalid sweep_id: {sweep_id}. Available sweeps: {list(SWEEPS.keys())}")
            return
        
        # Get model type from sweep
        model_type = SWEEPS[sweep_id]
        
        # Process the single run
        process_single_run(args, sweep_id, args.run_id, model_type, args.gpu_id)
    else:
        logger.error("No run_id specified and auto-detect not enabled. Nothing to do.")
        parser.print_help()

if __name__ == "__main__":
    main() 