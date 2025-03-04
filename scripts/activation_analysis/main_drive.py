"""
Activation Analysis with Google Drive Support

Main entry point for the activation analysis pipeline with Google Drive support.
This is a modified version of main.py that supports loading models from Google Drive while using local MSP data.
"""
import os
import torch
import pandas as pd
import logging
import traceback
from tqdm.auto import tqdm
from datetime import datetime
import json
import argparse
import numpy as np

from scripts.activation_analysis.config import *
from scripts.activation_analysis.utils import setup_logging, save_results_in_csv_format, is_s3_path, extract_checkpoint_number, ensure_dir, standardize, report_variance_explained
from scripts.activation_analysis.data_loading import ModelDataManager, ActivationExtractor
from scripts.activation_analysis.belief_states import BeliefStateGenerator
from scripts.activation_analysis.regression import RegressionAnalyzer, compute_efficient_pinv_from_svd
from epsilon_transformers.analysis.drive_loader import GoogleDriveModelLoader

# Whether to run classical belief analysis
RUN_CLASSICAL_BELIEFS = True

# Whether to run random baseline
DO_RANDOM_BASELINE = True

# Number of random runs for baseline
NUM_RANDOM_RUNS = 10

# Logger for this module
logger = logging.getLogger(__name__)

def process_checkpoint(checkpoint, sweep_id, run_id, model_type, 
                      model_data_manager, act_extractor, belief_analyzer, 
                      classical_beliefs, nn_beliefs, nn_inputs, nn_word_probs):
    """Process a single checkpoint."""
    checkpoint_number = extract_checkpoint_number(checkpoint)
    logger.info(f"Processing checkpoint {checkpoint_number}")
    
    # Load the checkpoint model
    ckpt_model, ckpt_config = model_data_manager.load_checkpoint(sweep_id, run_id, checkpoint)
    ckpt_model = ckpt_model.to(model_data_manager.device)
    
    # Prepare belief targets dictionary and corresponding activations
    belief_targets = {}
    target_activations = {}
    relevant_keys = TRANSFORMER_ACTIVATION_KEYS if model_type == 'transformer' else None
    
    # Extract activations for neural network beliefs
    nn_acts = act_extractor.extract_activations(ckpt_model, nn_inputs, model_type, relevant_keys)
    
    # Add neural network beliefs
    belief_targets['nn_beliefs'] = {
        'beliefs': nn_beliefs,
        'probs': nn_word_probs
    }
    target_activations['nn_beliefs'] = nn_acts
    
    # Add classical beliefs if available
    if classical_beliefs:
        for belief_type, belief_data in classical_beliefs.items():
            belief_targets[belief_type] = {
                'beliefs': belief_data['beliefs'],
                'probs': belief_data['probs']
            }
            target_activations[belief_type] = nn_acts  # Use same activations for classical beliefs
    
    # Run regression for each belief type
    regression_analyzer = RegressionAnalyzer()
    results = regression_analyzer.run_analysis(
        belief_targets=belief_targets,
        activations=target_activations,
        checkpoint=checkpoint_number
    )
    
    # Print summary of results
    logger.info("=" * 80)
    logger.info(f"REGRESSION RESULTS FOR CHECKPOINT {checkpoint_number}:")
    for belief_type, target_results in results.items():
        mses = [x['mse']['mean'] for x in target_results if 'mse' in x]
        if mses:
            logger.info(f"  {belief_type}: Average MSE = {np.mean(mses):.6f}")
    logger.info("=" * 80)
    
    return results

def process_run(sweep_id, run_id, model_type, device='cpu', output_dir=None, drive_path=None, use_local_msp=True):
    """Process a single run from a sweep.
    
    Args:
        sweep_id: ID of the sweep
        run_id: ID of the run
        model_type: Type of model (transformer or RNN)
        device: Device to use for computation
        output_dir: Directory to save results
        drive_path: Path to Google Drive directory containing models
        use_local_msp: Whether to use local MSP data storage instead of Google Drive
    
    Returns:
        Message indicating the status of processing
    """
    # Check if regression results already exist
    if output_dir:
        run_output_dir = os.path.join(output_dir, run_id)
        ensure_dir(run_output_dir)
        regression_output_path = os.path.join(run_output_dir, "regression_results.pkl")
        
        if os.path.exists(regression_output_path):
            logger.info(f"Regression results already exist for {run_id}. Skipping...")
            return f"Skipped run {run_id} - results already exist."
    
    # Initialize model data manager
    if drive_path:
        drive_loader = GoogleDriveModelLoader(base_drive_path=drive_path)
        # Use local MSP data even when loading models from Drive if requested
        model_data_manager = ModelDataManager(
            loader=drive_loader, 
            device=device,
            use_local_msp=use_local_msp  # This flag will ensure MSP data is stored locally
        )
    else:
        model_data_manager = ModelDataManager(device=device)
    
    # Prepare neural network belief states
    logger.info(f"Preparing neural network belief states...")
    act_extractor = ActivationExtractor(device=device)
    belief_analyzer = BeliefStateGenerator(model_data_manager, device=device)
    
    try:
        # Get available checkpoints
        checkpoints = model_data_manager.list_checkpoints(sweep_id, run_id)
        metadata = {
            'run_id': run_id,
            'sweep_id': sweep_id,
            'model_type': model_type,
            'device': device,
            'processing_time': str(datetime.now()),
            'status': 'in_progress'
        }
        
        if not checkpoints:
            logger.warning(f"No checkpoints found for run {run_id}.")
            metadata['status'] = 'no_checkpoints'
            with open(os.path.join(run_output_dir, f"{run_id}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            return f"Error: No checkpoints found for run {run_id}."
        
        # Sort checkpoints by number
        checkpoints.sort(key=extract_checkpoint_number)
        
        # Get the last checkpoint for analysis
        checkpoint = checkpoints[-1]
        logger.info(f"Using checkpoint: {checkpoint}")
        
        # Load run configuration
        _, run_config = model_data_manager.load_checkpoint(sweep_id, run_id, checkpoint)
        metadata['run_config'] = run_config
        
        # Load loss data if available
        try:
            loss_df = model_data_manager.load_loss_from_run(sweep_id, run_id)
            if loss_df is not None:
                logger.info(f"Loaded loss data with {len(loss_df)} entries.")
                metadata['loss_loaded'] = True
                
                # Save loss data to results directory
                loss_file_path = os.path.join(run_output_dir, f"{run_id}_loss.csv")
                loss_df.to_csv(loss_file_path, index=False)
                metadata['loss_file'] = loss_file_path
        except Exception as e:
            logger.warning(f"Could not load loss data: {e}")
            metadata['loss_loaded'] = False
        
        # Process neural network beliefs
        logger.info("Preparing neural network beliefs...")
        # Use prepare_msp_data from drive_patches if using Google Drive
        if drive_path:
            try:
                from scripts.activation_analysis.drive_patches import prepare_msp_data
                nn_inputs, nn_beliefs, nn_indices, nn_word_probs, _ = prepare_msp_data(
                    run_config, run_config.get("model_config", {}), loader=drive_loader
                )
            except Exception as e:
                logger.error(f"Error preparing MSP data with Google Drive loader: {e}")
                from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
                nn_inputs, nn_beliefs, nn_indices, nn_word_probs, _ = prepare_msp_data(
                    run_config, run_config.get("model_config", {})
                )
        else:
            from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
            nn_inputs, nn_beliefs, nn_indices, nn_word_probs, _ = prepare_msp_data(
                run_config, run_config.get("model_config", {})
            )
        
        # Generate classical belief states if needed
        classical_beliefs = None
        if RUN_CLASSICAL_BELIEFS:
            logger.info("Generating classical belief states...")
            classical_beliefs = belief_analyzer.generate_classical_belief_states(
                run_config, max_order=MAX_MARKOV_ORDER
            )
            
            # Display a summary of the generated classical beliefs
            if classical_beliefs:
                logger.info("=" * 80)
                logger.info("CLASSICAL BELIEF STATES SUMMARY:")
                for key in sorted(classical_beliefs.keys()):
                    belief_shape = classical_beliefs[key]['beliefs'].shape
                    logger.info(f"  {key}: shape = {belief_shape}, dimension = {belief_shape[-1]}")
                    if belief_shape[-1] >= 64:
                        logger.info(f"  NOTE: {key} has reached or exceeded the dimension limit of 64")
                logger.info("=" * 80)
            else:
                logger.warning("No classical belief states were generated!")
        
        # MEMORY-EFFICIENT VERSION: Process random baselines one at a time
        random_data = None
        if DO_RANDOM_BASELINE:
            relevant_keys = TRANSFORMER_ACTIVATION_KEYS if model_type == 'transformer' else None
            
            # Prepare belief targets for random baseline
            baseline_targets = {}
            baseline_targets['nn_beliefs'] = {
                'beliefs': nn_beliefs,
                'probs': nn_word_probs
            }
            if classical_beliefs is not None:
                for mk in classical_beliefs:
                    baseline_targets[mk] = {
                        'beliefs': classical_beliefs[mk]['beliefs'],
                        'probs': classical_beliefs[mk]['probs']
                    }
            
            logger.info(f"Running memory-efficient random baseline analysis for {len(baseline_targets)} target types")
            
            random_analyzer = RegressionAnalyzer()
            random_data = random_analyzer.run_random_baseline(
                belief_targets=baseline_targets,
                activation_shape=act_extractor.get_activation_shape(model_type, nn_inputs),
                relevant_keys=relevant_keys,
                num_random_runs=NUM_RANDOM_RUNS
            )
            logger.info("Completed random baseline analysis")
        
        # Process the checkpoint
        checkpoint_data = process_checkpoint(
            checkpoint=checkpoint,
            sweep_id=sweep_id,
            run_id=run_id,
            model_type=model_type,
            model_data_manager=model_data_manager,
            act_extractor=act_extractor,
            belief_analyzer=belief_analyzer,
            classical_beliefs=classical_beliefs,
            nn_beliefs=nn_beliefs,
            nn_inputs=nn_inputs,
            nn_word_probs=nn_word_probs
        )
        
        # Save the results
        logger.info(f"Saving results for run {run_id} to {run_output_dir}")
        
        # Prepare checkpoint data structure
        regression_results = []
        for belief_type, target_results in checkpoint_data.items():
            for layer_result in target_results:
                if 'mse' in layer_result:  # Only include results with MSE data
                    row = {
                        'run_id': run_id,
                        'checkpoint': layer_result.get('checkpoint', 'unknown'),
                        'belief_type': belief_type,
                        'layer': layer_result.get('layer', 'unknown'),
                        'mse': layer_result['mse'].get('mean', 0),
                        'mse_std': layer_result['mse'].get('std', 0),
                        'r2': layer_result.get('r2', 0),
                        'dimension': layer_result.get('dimension', 0),
                        'samples': layer_result.get('samples', 0)
                    }
                    regression_results.append(row)
        
        # Save to CSV
        if regression_results:
            df = pd.DataFrame(regression_results)
            df.to_csv(os.path.join(run_output_dir, f"{run_id}_regression_results.csv"), index=False)
            metadata['regression_results'] = os.path.join(run_output_dir, f"{run_id}_regression_results.csv")
            logger.info(f"Saved {len(regression_results)} regression results to {metadata['regression_results']}")
        
        # Add random baseline data if available
        if random_data:
            random_rows = []
            for belief_type, target_results in random_data.items():
                for layer_result in target_results:
                    if 'random_mse' in layer_result:  # Only include results with random MSE data
                        row = {
                            'run_id': run_id,
                            'belief_type': belief_type,
                            'layer': layer_result.get('layer', 'unknown'),
                            'random_mse': layer_result['random_mse'].get('mean', 0),
                            'random_mse_std': layer_result['random_mse'].get('std', 0),
                            'dimension': layer_result.get('dimension', 0),
                            'samples': layer_result.get('samples', 0),
                            'iterations': layer_result.get('iterations', 0)
                        }
                        random_rows.append(row)
            
            if random_rows:
                random_df = pd.DataFrame(random_rows)
                random_path = os.path.join(run_output_dir, f"{run_id}_random_baseline.csv")
                random_df.to_csv(random_path, index=False)
                metadata['random_baseline'] = random_path
                logger.info(f"Saved {len(random_rows)} random baseline results to {metadata['random_baseline']}")
        
        # Mark the run as completed
        metadata['status'] = 'completed'
        metadata['completion_time'] = str(datetime.now())
        
        # Save metadata
        with open(os.path.join(run_output_dir, f"{run_id}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
    except Exception as e:
        logger.error(f"Error processing run {run_id}: {e}")
        logger.error(traceback.format_exc())
        
        # Update metadata with error info
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        metadata['traceback'] = traceback.format_exc()
        
        # Save metadata even if there was an error
        try:
            with open(os.path.join(run_output_dir, f"{run_id}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e2:
            logger.error(f"Error saving metadata after run failure: {e2}")
        
        return f"Error processing run {run_id}: {e}"
    
    # Try to update metadata with the path to the loss file
    try:
        if 'loss_file' in metadata:
            with open(os.path.join(run_output_dir, f"{run_id}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error updating metadata with loss file path: {e}")

    return f"Processed run {run_id} (model: {model_type}) on device {device}."

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run activation analysis pipeline with Google Drive support')
    parser.add_argument('--output-dir', type=str, default=None,
                      help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--s3-output', type=str, default=None,
                      help='S3 output path (e.g., s3://bucket-name/path/to/output)')
    parser.add_argument('--drive-path', type=str, default=None,
                      help='Google Drive path for models and data (e.g., /content/drive/My Drive/quantum/)')
    parser.add_argument('--sweep-id', type=str, default=None,
                      help='Process only runs from this sweep (default: auto-detect all sweeps)')
    parser.add_argument('--run-id', type=str, default=None,
                      help='Process only this specific run (default: all runs)')
    parser.add_argument('--device', type=str, default=None,
                      help=f'Device to use (default: {DEFAULT_DEVICE})')
    parser.add_argument('--use-local-msp', action='store_true', default=True,
                      help='Use local MSP data storage instead of Google Drive (default: True)')
    parser.add_argument('--layer-filter', type=str, default='L4',
                      help='Filter runs by layer pattern (default: L4)')
    args = parser.parse_args()
    
    # Determine output directory (local, S3, or Drive)
    output_dir = None
    if args.s3_output:
        if not args.s3_output.startswith('s3://'):
            logger.warning(f"S3 output path should start with 's3://'. Got: {args.s3_output}")
            logger.warning(f"Prepending 's3://' to the path")
            output_dir = f"s3://{args.s3_output}"
        else:
            output_dir = args.s3_output
        logger.info(f"Using S3 output directory: {output_dir}")
    elif args.output_dir:
        output_dir = args.output_dir
        logger.info(f"Using custom output directory: {output_dir}")
    elif args.drive_path:
        # If drive_path is specified but no output_dir, use a subdirectory in the drive path
        output_dir = os.path.join(args.drive_path, "analysis")
        logger.info(f"Using Google Drive output directory: {output_dir}")
    
    # Set up logging with a directory based on OUTPUT_DIR
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger.info(f"Starting activation analysis pipeline. Log file: {log_file}")
    
    # Set device to use
    device = args.device or DEFAULT_DEVICE
    
    # Process each sweep
    sweeps_to_process = SWEEPS
    
    # If using Google Drive, list available sweeps
    if args.drive_path:
        logger.info(f"Initializing Google Drive loader to list available sweeps")
        drive_loader = GoogleDriveModelLoader(base_drive_path=args.drive_path)
        available_sweeps = drive_loader.list_sweeps()
        logger.info(f"Found {len(available_sweeps)} sweeps in Google Drive: {available_sweeps}")
        
        if args.sweep_id:
            # Use specified sweep if it exists
            if args.sweep_id in available_sweeps:
                logger.info(f"Using specified sweep: {args.sweep_id}")
                available_sweeps = [args.sweep_id]
            else:
                logger.error(f"Specified sweep '{args.sweep_id}' not found in Google Drive")
                return
        
        # Check which sweeps are in the config
        config_sweeps = set(SWEEPS.keys())
        drive_sweeps = set(available_sweeps)
        common_sweeps = config_sweeps.intersection(drive_sweeps)
        
        if common_sweeps:
            logger.info(f"Processing {len(common_sweeps)} sweeps that exist in both config and Google Drive")
            sweeps_to_process = {s: SWEEPS[s] for s in common_sweeps}
        else:
            # If no common sweeps, try to infer model types for available sweeps
            logger.warning("No sweeps from the config found in Google Drive")
            logger.info("Attempting to infer model types from sweep names")
            
            inferred_sweeps = {}
            for sweep in available_sweeps:
                # Simple heuristic: guess model type based on sweep name
                if "transformer" in sweep.lower():
                    model_type = "transformer"
                elif "rnn" in sweep.lower():
                    model_type = "rnn"
                else:
                    # Default to transformer if can't determine
                    model_type = "transformer"
                    
                inferred_sweeps[sweep] = model_type
                logger.info(f"Inferred model type for sweep '{sweep}': {model_type}")
            
            sweeps_to_process = inferred_sweeps
    elif args.sweep_id:
        # If not using Google Drive but sweep_id is specified
        if args.sweep_id in SWEEPS:
            sweeps_to_process = {args.sweep_id: SWEEPS[args.sweep_id]}
        else:
            logger.error(f"Sweep ID '{args.sweep_id}' not found in config.")
            return
    
    for sweep_id, model_type in sweeps_to_process.items():
        logger.info(f"Processing sweep {sweep_id} ({model_type})")
        
        # Initialize data manager (either with Google Drive or default S3)
        if args.drive_path:
            drive_loader = GoogleDriveModelLoader(base_drive_path=args.drive_path)
            model_data_manager = ModelDataManager(
                loader=drive_loader, 
                device=device,
                use_local_msp=args.use_local_msp
            )
        else:
            model_data_manager = ModelDataManager(device=device)
        
        # Get runs to process
        if args.run_id:
            runs = [args.run_id]
            logger.info(f"Processing single run: {args.run_id}")
        else:
            # Get all runs in the sweep, filter by layer pattern if specified
            layer_filter = args.layer_filter
            all_runs = model_data_manager.list_runs_in_sweep(sweep_id)
            
            if layer_filter:
                runs = [run_id for run_id in all_runs if layer_filter in run_id]
                logger.info(f"Found {len(runs)} runs matching filter '{layer_filter}' in sweep {sweep_id}")
            else:
                runs = all_runs
                logger.info(f"Found {len(runs)} total runs in sweep {sweep_id}")
        
        # Process each run
        for run_id in runs:
            try:
                sweep_output_dir = output_dir
                if is_s3_path(output_dir):
                    # For S3 paths, we want to preserve the structure
                    sweep_output_dir = f"{output_dir}/{sweep_id}"
                
                result = process_run(
                    sweep_id=sweep_id,
                    run_id=run_id,
                    model_type=model_type,
                    device=device,
                    output_dir=sweep_output_dir,
                    drive_path=args.drive_path,
                    use_local_msp=args.use_local_msp
                )
                logger.info(result)
            except Exception as e:
                logger.error(f"Error processing run {run_id}: {e}")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 