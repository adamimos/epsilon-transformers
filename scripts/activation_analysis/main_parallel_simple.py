"""
Simplified parallel execution for activation analysis.
This script combines run detection, distribution, and processing in one file.
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

from config import (
    OUTPUT_DIR, SWEEPS, RCOND_SWEEP_LIST, 
    MAX_MARKOV_ORDER, TRANSFORMER_ACTIVATION_KEYS,
    MAX_CHECKPOINTS, PROCESS_ALL_CHECKPOINTS
)
from main import process_run
from utils import setup_logging

# Set up module logger
logger = logging.getLogger("main_parallel")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run activation analysis in parallel')
    
    # Run selection (mutually exclusive)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument('--run-id', type=str, default=None,
                      help='Specific run ID to process (default: auto-detect runs)')
    run_group.add_argument('--auto-detect', action='store_true',
                      help='Auto-detect available runs')
    run_group.add_argument('--run-list', type=str, default=None,
                      help='JSON file with list of run IDs to process')
    
    # Parallelization options
    parser.add_argument('--num-gpus', type=int, default=1,
                      help='Number of GPUs to distribute jobs across (default: 1)')
    parser.add_argument('--gpu-id', type=int, default=0,
                      help='Specific GPU ID to use when processing a single run (default: 0)')
    
    # Run filtering
    parser.add_argument('--sweep-id', type=str, default=None,
                      help='Process only runs from this sweep (default: all sweeps)')
    
    # Processing options
    parser.add_argument('--output-dir', type=str, default=None,
                      help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--log-dir', type=str, default=None,
                      help='Log directory (defaults to OUTPUT_DIR/logs)')
    parser.add_argument('--max-checkpoints', type=int, default=None,
                      help=f'Maximum number of checkpoints to process (default: {MAX_CHECKPOINTS})')
    parser.add_argument('--process-all-checkpoints', action='store_true',
                      help='Process all checkpoints, not just the most recent ones')
    
    # Execution options
    parser.add_argument('--dry-run', action='store_true',
                      help='Print commands without executing them')
    parser.add_argument('--wait', action='store_true',
                      help='Wait for each batch to complete before starting the next')
    parser.add_argument('--generate-vast-ai', action='store_true',
                      help='Generate vast.ai compatible shell scripts instead of running directly')
    
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
    try:
        from data_loading import ModelDataManager
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

def process_single_run(args, sweep_id, run_id, model_type, gpu_id=0):
    """Process a single run on the specified GPU."""
    try:
        device = setup_device(gpu_id)
        logger.info(f"Processing run {run_id} from sweep {sweep_id} ({model_type}) on device {device}")
        result = process_run(sweep_id, run_id, model_type, device=device)
        logger.info(result)
        return True
    except Exception as e:
        logger.error(f"Error processing run {run_id} in sweep {sweep_id}:")
        logger.error(traceback.format_exc())
        return False

def generate_vast_ai_scripts(run_list, args):
    """Generate shell scripts for vast.ai execution."""
    # Distribute runs evenly across GPUs
    gpu_runs = defaultdict(list)
    for i, run_id in enumerate(run_list):
        gpu_id = i % args.num_gpus
        gpu_runs[gpu_id].append(run_id)
    
    # Create output directory for scripts
    scripts_dir = os.path.join(args.output_dir or OUTPUT_DIR, "vast_ai_scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    logger.info(f"Generating vast.ai scripts in {scripts_dir}")
    
    # Generate a script for each GPU
    for gpu_id, runs in gpu_runs.items():
        script_path = os.path.join(scripts_dir, f"run_gpu_{gpu_id}.sh")
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("# This script was auto-generated for vast.ai\n\n")
            f.write("# Clone the repository if not already present\n")
            f.write("if [ ! -d \"epsilon-transformers\" ]; then\n")
            f.write("  git clone https://github.com/yourusername/epsilon-transformers.git\n")
            f.write("  cd epsilon-transformers\n")
            f.write("else\n")
            f.write("  cd epsilon-transformers\n")
            f.write("  git pull\n")
            f.write("fi\n\n")
            f.write("# Install requirements\n")
            f.write("pip install -r requirements.txt\n\n")
            f.write("# Run the analysis for each assigned run_id\n")
            
            for run_id in runs:
                # Determine sweep_id if needed
                auto_sweep_id = None
                for sid in SWEEPS:
                    if sid in run_id:
                        auto_sweep_id = sid
                        break
                
                cmd = ["python", "-m", "scripts.activation_analysis.main_parallel_simple"]
                cmd.extend(["--run-id", run_id])
                cmd.extend(["--gpu-id", "0"])  # Use GPU 0 within the container
                
                if auto_sweep_id:
                    cmd.extend(["--sweep-id", auto_sweep_id])
                
                if args.output_dir:
                    cmd.extend(["--output-dir", args.output_dir])
                
                if args.max_checkpoints is not None:
                    cmd.extend(["--max-checkpoints", str(args.max_checkpoints)])
                
                if args.process_all_checkpoints:
                    cmd.append("--process-all-checkpoints")
                
                f.write(" ".join(str(c) for c in cmd) + "\n")
        
        # Make script executable
        os.chmod(script_path, 0o755)
        logger.info(f"Generated script for GPU {gpu_id}: {script_path}")
    
    # Generate README with instructions
    readme_path = os.path.join(scripts_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# Instructions for Running on vast.ai\n\n")
        f.write("## Overview\n\n")
        f.write("These scripts distribute the activation analysis workload across multiple ")
        f.write("machines on vast.ai. Each script handles a subset of run_ids.\n\n")
        
        f.write("## Steps to run on vast.ai\n\n")
        f.write("1. Upload these scripts to a location accessible by your vast.ai instances\n")
        f.write("2. For each script, create a separate vast.ai instance\n")
        f.write("3. When starting each instance, use the following configuration:\n\n")
        
        f.write("```bash\n")
        f.write("# In the vast.ai interface, under 'On-start script', use:\n")
        f.write("cd /path/to/scripts && bash run_gpu_X.sh\n")
        f.write("```\n\n")
        
        f.write("Replace `X` with the appropriate GPU number and `/path/to/scripts` ")
        f.write("with the location where you uploaded the scripts.\n")
    
    logger.info(f"Generated vast.ai instructions: {readme_path}")
    return scripts_dir

def distribute_runs(run_list, args):
    """Distribute runs across multiple GPUs."""
    if not run_list:
        logger.warning("No runs to process")
        return
    
    # If generating vast.ai scripts, delegate to that function
    if args.generate_vast_ai:
        return generate_vast_ai_scripts(run_list, args)
    
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
                    cmd = ["python", "-m", "scripts.activation_analysis.main_parallel_simple"]
                    cmd.extend(["--run-id", run_id])
                    cmd.extend(["--gpu-id", str(gpu_id)])
                    
                    if sweep_id:
                        cmd.extend(["--sweep-id", sweep_id])
                    
                    if args.output_dir:
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
    args = parse_arguments()
    
    # Set up output directory
    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_dir = args.log_dir or os.path.join(output_dir, "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = setup_logging(log_dir=log_dir)
    
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
    if args.run_id:
        # Single run mode
        logger.info(f"Starting parallel activation analysis for run {args.run_id}. Log file: {log_file}")
        
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
        # Multiple run mode
        logger.info(f"Starting multi-run parallel activation analysis. Log file: {log_file}")
        
        # Get list of runs to process
        run_list = get_run_list(args)
        logger.info(f"Found {len(run_list)} runs to process")
        
        # Distribute runs
        distribute_runs(run_list, args)

if __name__ == "__main__":
    main() 