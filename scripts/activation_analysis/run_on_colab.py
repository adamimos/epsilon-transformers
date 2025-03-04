#!/usr/bin/env python
"""
Script for running activation analysis in Google Colab with Google Drive integration.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(repo_root))

# Import required modules
from scripts.activation_analysis.config import *
from scripts.activation_analysis.utils import setup_logging
from scripts.activation_analysis.data_loading import ModelDataManager, ActivationExtractor
from scripts.activation_analysis.belief_states import BeliefStateGenerator
from scripts.activation_analysis.regression import RegressionAnalyzer
from epsilon_transformers.analysis.drive_loader import GoogleDriveModelLoader

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run activation analysis on Colab with Google Drive')
    parser.add_argument('--drive-path', type=str, default='/content/drive/My Drive/quantum/',
                      help='Base path in Google Drive where data is stored')
    parser.add_argument('--sweep-id', type=str, required=True,
                      help='Process only runs from this sweep')
    parser.add_argument('--run-id', type=str, default=None,
                      help='Process only this specific run (if not specified, all runs in sweep will be processed)')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for analysis results (defaults to drive-path/analysis)')
    return parser.parse_args()

def run_analysis_with_drive(args):
    """Run activation analysis using Google Drive loader."""
    # Set up logging
    log_dir = os.path.join(args.drive_path, "analysis", "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger = logging.getLogger("main")
    logger.info(f"Starting activation analysis with Google Drive. Log file: {log_file}")
    
    # Create the Google Drive model loader
    loader = GoogleDriveModelLoader(base_drive_path=args.drive_path)
    logger.info(f"Initialized Google Drive loader with base path: {args.drive_path}")
    
    # Set output directory
    output_dir = args.output_dir or os.path.join(args.drive_path, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    
    # Initialize data manager with the Google Drive loader
    model_data_manager = ModelDataManager(loader=loader, device=args.device)
    logger.info(f"Initialized model data manager on device: {args.device}")
    
    # Get sweep info from config or fallback to guessing based on sweep_id naming
    sweep_id = args.sweep_id
    if sweep_id in SWEEPS:
        model_type = SWEEPS[sweep_id]
        logger.info(f"Found sweep {sweep_id} in config with model type: {model_type}")
    else:
        # Guess model type based on sweep_id naming convention
        model_type = 'transformer' if 'transformer' in sweep_id.lower() else 'rnn'
        logger.warning(f"Sweep {sweep_id} not found in config. Guessing model type: {model_type}")
    
    # Determine which runs to process
    if args.run_id:
        runs = [args.run_id]
        logger.info(f"Processing single run: {args.run_id}")
    else:
        # Get all runs in the sweep
        runs = model_data_manager.list_runs_in_sweep(sweep_id)
        logger.info(f"Found {len(runs)} runs in sweep {sweep_id}")
    
    # Process each run
    for run_id in runs:
        logger.info(f"Processing run {run_id} from sweep {sweep_id}")
        
        try:
            # List available checkpoints
            checkpoints = model_data_manager.list_checkpoints(sweep_id, run_id)
            if not checkpoints:
                logger.warning(f"No checkpoints found for run {run_id}. Skipping.")
                continue
            
            # Sort checkpoints (assuming standard naming like 'model_1000.pt')
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)
            
            # Use the latest checkpoint by default
            checkpoint_key = checkpoints[-1]
            logger.info(f"Using checkpoint: {checkpoint_key}")
            
            # Load the model and config
            model, run_config = model_data_manager.load_checkpoint(
                sweep_id, run_id, checkpoint_key, device=args.device
            )
            logger.info(f"Successfully loaded model from checkpoint {checkpoint_key}")
            
            # Create activation extractor and belief state generator
            act_extractor = ActivationExtractor(device=args.device)
            belief_analyzer = BeliefStateGenerator(device=args.device)
            
            # Prepare process configurations
            process_config = run_config.get('process_config', {})
            if not process_config:
                logger.warning("Process config not found in run config. Analysis may fail.")
            
            # Run the analysis
            from scripts.activation_analysis.main import process_checkpoint
            
            # Generate or load beliefs based on process configuration
            from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
            nn_inputs, nn_beliefs, nn_indices, nn_word_probs, nn_unnormalized = prepare_msp_data(
                run_config, run_config.get("model_config", {}), loader=loader
            )
            
            # Run the process checkpoint analysis
            result = process_checkpoint(
                checkpoint=checkpoint_key,
                sweep_id=sweep_id,
                run_id=run_id,
                model_type=model_type,
                model_data_manager=model_data_manager,
                act_extractor=act_extractor,
                belief_analyzer=belief_analyzer,
                classical_beliefs=None,  # Optional classical beliefs
                nn_beliefs=nn_beliefs,
                nn_inputs=nn_inputs,
                nn_word_probs=nn_word_probs
            )
            
            logger.info(f"Analysis completed for run {run_id}: {result}")
            
        except Exception as e:
            logger.error(f"Error processing run {run_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("Activation analysis completed successfully")

def main():
    """Main entry point."""
    args = parse_args()
    run_analysis_with_drive(args)

if __name__ == "__main__":
    main()
