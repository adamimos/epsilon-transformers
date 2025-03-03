"""
Main entry point for the activation analysis pipeline.
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
from scripts.activation_analysis.utils import setup_logging, save_results_in_csv_format, is_s3_path, extract_checkpoint_number, ensure_dir
from scripts.activation_analysis.data_loading import ModelDataManager, ActivationExtractor
from scripts.activation_analysis.belief_states import BeliefStateGenerator
from scripts.activation_analysis.regression import RegressionAnalyzer

# Set up module logger
logger = logging.getLogger("main")

# Define additional constants
RUN_CLASSICAL_BELIEFS = True  # Default value
DO_RANDOM_BASELINE = True  # Default value

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
    
    # Add classical beliefs - extract activations using the correct inputs for each
    for mk in classical_beliefs:
        # Use the markov-specific inputs to extract activations
        mk_inputs = classical_beliefs[mk]['inputs']
        mk_acts = act_extractor.extract_activations(ckpt_model, mk_inputs, model_type, relevant_keys)
        
        belief_targets[mk] = {
            'beliefs': classical_beliefs[mk]['beliefs'],
            'probs': classical_beliefs[mk]['probs']
        }
        target_activations[mk] = mk_acts
    
    # Analyze the checkpoint for all targets and regularization parameters
    results_df, best_weights, best_singular = belief_analyzer.analyze_checkpoint_with_activations(
        target_activations, belief_targets, RCOND_SWEEP_LIST
    )
    
    # Add checkpoint information to the results
    results_df['checkpoint'] = checkpoint_number
    results_df['sweep_id'] = sweep_id
    results_df['run_id'] = run_id
    
    return {
        'checkpoint': checkpoint_number,
        'results_df': results_df,
        'best_weights': best_weights,
        'best_singular': best_singular
    }

def process_run(sweep_id, run_id, model_type, device='cpu', output_dir=None):
    """Process a single run."""
    # Only process runs with 'L4' in run_id (4-layer networks)
    if 'L4' not in run_id:
        logger.info(f"Skipping run {run_id} (not a 4-layer network).")
        return f"Skipped run {run_id} (not a 4-layer network)."

    # Setup output directory
    base_outdir = output_dir if output_dir is not None else os.path.join(OUTPUT_DIR, sweep_id)
    ensure_dir(base_outdir)

    # Check if this run has already been processed
    run_dir = os.path.join(base_outdir, run_id)
    metadata_path = os.path.join(run_dir, f"{run_id}_metadata.json")
    regression_results_path = os.path.join(run_dir, f"{run_id}_regression_results.csv")
    
    if os.path.exists(metadata_path) and os.path.exists(regression_results_path):
        logger.info(f"Skipping run {run_id} as it has already been processed (output files exist).")
        return f"Skipped run {run_id} (already processed)."

    logger.info(f"Processing run {run_id} (model type: {model_type}) on device {device}")

    # Initialize components
    model_data_manager = ModelDataManager(device=device)
    act_extractor = ActivationExtractor(device=device)
    belief_generator = BeliefStateGenerator(model_data_manager, device=device)
    regression_analyzer = RegressionAnalyzer(device=device,
                                             use_efficient_pinv=True)

    # Load an initial checkpoint to get configuration
    all_checkpoints = model_data_manager.list_checkpoints(sweep_id, run_id)
    if not all_checkpoints:
        logger.warning(f"No checkpoints found for run {run_id}")
        return f"No checkpoints found for run {run_id}"
        
    init_checkpoint = all_checkpoints[0]
    model, run_config = model_data_manager.load_checkpoint(sweep_id, run_id, init_checkpoint)
    run_config["sweep_id"] = sweep_id  # store sweep id for later use
    model = model.to(device)

    # Load model inputs and network belief states
    logger.info("Loading mixed state presentation data")
    nn_inputs, nn_beliefs, _, nn_word_probs, _ = model_data_manager.load_msp_data(run_config)
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_word_probs = nn_word_probs.to(device)

    # Generate classical belief states
    classical_beliefs = None
    if RUN_CLASSICAL_BELIEFS:
        logger.info("Generating classical belief states with max order %d...", MAX_MARKOV_ORDER)
        classical_beliefs = belief_generator.generate_classical_belief_states(
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
        
        # Run random baseline analysis with rcond sweep
        all_baseline_results = []
        all_baseline_singular_values = {}
        all_baseline_weights = {}
        
        # Restructured to match trained network processing:
        # network -> target -> rcond
        for random_idx in range(NUM_RANDOM_BASELINES):
            logger.info(f"Processing random network {random_idx}")
            random_activations = {}
            
            # First, extract activations for all targets
            try:
                # Extract nn_beliefs activations
                nn_generator = act_extractor.get_random_activations_streaming(
                    model, run_config, nn_inputs, model_type, relevant_keys,
                    num_baselines=1,
                    seed=random_idx
                )
                random_seed, nn_acts = next(nn_generator, (None, None))
                if random_seed is None:
                    logger.warning(f"Failed to generate random network at index {random_idx}")
                    continue
                    
                random_activations['nn_beliefs'] = nn_acts
                
                # Extract classical belief activations if available
                if classical_beliefs is not None:
                    for mk in classical_beliefs:
                        mk_inputs = classical_beliefs[mk]['inputs']
                        mk_generator = act_extractor.get_random_activations_streaming(
                            model, run_config, mk_inputs, model_type, relevant_keys,
                            num_baselines=1,
                            seed=random_idx
                        )
                        _, mk_acts = next(mk_generator, (None, None))
                        random_activations[mk] = mk_acts
                
                # Now process all targets for this random network
                for target_name, target_activations in random_activations.items():
                    target_beliefs = None
                    target_probs = None
                    
                    if target_name == 'nn_beliefs':
                        target_beliefs = nn_beliefs
                        target_probs = nn_word_probs
                    elif classical_beliefs is not None and target_name in classical_beliefs:
                        target_beliefs = classical_beliefs[target_name]['beliefs']
                        target_probs = classical_beliefs[target_name]['probs']
                    
                    if target_beliefs is not None:
                        # Process this target with all rcond values
                        for rcond_val in tqdm(RCOND_SWEEP_LIST, 
                                             desc=f"Processing rcond values for network {random_idx}, target {target_name}"):
                            df, singular_values, weights, _, best_layers = regression_analyzer.process_activation_layers(
                                target_activations, target_beliefs, target_probs, rcond_val
                            )
                            
                            # Add metadata
                            df['checkpoint'] = f'RANDOM_{random_idx}'
                            df['target'] = target_name
                            df['rcond'] = rcond_val
                            all_baseline_results.append(df)
                            
                            # Store singular values
                            if target_name not in all_baseline_singular_values:
                                all_baseline_singular_values[target_name] = {}
                            
                            for layer, sv in singular_values.items():
                                if layer not in all_baseline_singular_values[target_name]:
                                    all_baseline_singular_values[target_name][layer] = []
                                
                                all_baseline_singular_values[target_name][layer].append({
                                    "random_idx": random_idx,
                                    "singular_values": sv
                                })
                            
                            # Store weights - only for lowest rcond
                            if rcond_val == min(RCOND_SWEEP_LIST):
                                if target_name not in all_baseline_weights:
                                    all_baseline_weights[target_name] = {}
                                
                                for layer, dist in best_layers.items():
                                    layer_key = f"{layer}_random_{random_idx}"
                                    
                                    all_baseline_weights[target_name][layer_key] = {
                                        "weights": weights.get(layer, None),
                                        "rcond": rcond_val,
                                        "dist": dist,
                                        "random_idx": random_idx
                                    }
                
                # Free memory explicitly
                del random_activations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing random network {random_idx}: {e}")
                logger.error(traceback.format_exc())
        
        # Combine results into final dataframe
        if all_baseline_results:
            all_df = pd.concat(all_baseline_results, ignore_index=True)
            random_data = {
                'df': all_df,
                'weights': all_baseline_weights,
                'singular': all_baseline_singular_values
            }
            logger.info(f"Random baseline analysis complete: {len(all_baseline_results)} results")
        else:
            logger.warning("No random baseline results were generated")
    
    # Process checkpoints
    if not PROCESS_ALL_CHECKPOINTS:
        checkpoints_to_process = all_checkpoints[:MAX_CHECKPOINTS]
    else:
        checkpoints_to_process = all_checkpoints
    
    logger.info(f"Processing {len(checkpoints_to_process)} checkpoints")
    
    all_checkpoint_results = []
    for checkpoint in tqdm(checkpoints_to_process, desc=f"Processing checkpoints for {run_id}"):
        try:
            result = process_checkpoint(
                checkpoint, 
                sweep_id, 
                run_id, 
                model_type,
                model_data_manager,
                act_extractor,
                regression_analyzer,
                classical_beliefs,
                nn_beliefs,
                nn_inputs,
                nn_word_probs
            )
            all_checkpoint_results.append(result)
        except Exception as e:
            logger.error(f"Error processing checkpoint {checkpoint}:")
            logger.error(traceback.format_exc())
    
    # Combine results from all checkpoints
    if all_checkpoint_results:
        # Combine DataFrames
        all_dfs = [r['results_df'] for r in all_checkpoint_results]
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Store best weights per-checkpoint
        # Organize as target -> layer -> checkpoint -> weight data
        combined_weights = {}
        
        # Find the best rcond value for each checkpoint-target-layer combination
        for result in all_checkpoint_results:
            checkpoint = result['checkpoint']
            
            # For each target in this checkpoint's results
            for target, target_weights in result['best_weights'].items():
                if target not in combined_weights:
                    combined_weights[target] = {}
                
                # For each layer in this target
                for layer, layer_data in target_weights.items():
                    if layer not in combined_weights[target]:
                        combined_weights[target][layer] = {}
                    
                    # Store checkpoint's best weights
                    layer_data['checkpoint'] = checkpoint
                    combined_weights[target][layer][checkpoint] = layer_data
        
        # Combine singular values
        combined_singular = {}
        for result in all_checkpoint_results:
            checkpoint = result['checkpoint']
            
            # Process singular values
            for target, target_sv in result['best_singular'].items():
                if target not in combined_singular:
                    combined_singular[target] = {}
                
                for layer, layer_sv_list in target_sv.items():
                    if layer not in combined_singular[target]:
                        combined_singular[target][layer] = []
                    
                    for sv_data in layer_sv_list:
                        sv_data['checkpoint'] = checkpoint
                        combined_singular[target][layer].append(sv_data)
        
        # Flatten weights structure for saving
        # We need to convert from target -> layer -> checkpoint -> data
        # to target -> layer -> data with checkpoint included
        flat_weights = {}
        for target, layers in combined_weights.items():
            flat_weights[target] = {}
            for layer, checkpoints in layers.items():
                for checkpoint, weight_data in checkpoints.items():
                    # Don't append checkpoint to layer name, keep layer identifiers clean
                    if layer not in flat_weights[target]:
                        flat_weights[target][layer] = {}
                    flat_weights[target][layer][checkpoint] = weight_data
        
        # Save results in CSV+NPY format
        saved_files = save_results_in_csv_format(
            output_dir=base_outdir,
            run_id=run_id,
            checkpoint_data={
                'df': combined_df.drop(['run_id', 'sweep_id', 'variance_explained'], axis=1, errors='ignore'),
                'weights': flat_weights,
                'singular': combined_singular,
                'attrs': {
                    "format_version": "csv_v1",
                    "creation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": run_id,
                    "sweep_id": sweep_id
                }
            },
            random_data=random_data if random_data is None else {
                'df': random_data['df'].drop(['run_id', 'sweep_id', 'variance_explained'], axis=1, errors='ignore'),
                'weights': random_data.get('weights', {}),
                'singular': random_data.get('singular', {})
            }
        )
        logger.info(f"Saved results to {base_outdir} (metadata: {saved_files['metadata']})")
    
    # Save loss data if available
    loss_df = model_data_manager.load_loss_from_run(sweep_id=sweep_id, run_id=run_id)
    if loss_df is not None:
        # Create run-specific directory for loss data (same as other files)
        run_dir = os.path.join(base_outdir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save loss data in the run-specific directory
        loss_csv = os.path.join(run_dir, f"{run_id}_loss.csv")
        loss_df.to_csv(loss_csv, index=False)
        logger.info(f"Saved loss data to {loss_csv}")
        
        # Update metadata to include the loss file path
        try:
            metadata_path = os.path.join(run_dir, f"{run_id}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Add loss file to file index
                if 'file_index' in metadata:
                    metadata['file_index']['loss_csv'] = loss_csv
                    
                    # Save updated metadata
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating metadata with loss file path: {e}")

    return f"Processed run {run_id} (model: {model_type}) on device {device}."

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run activation analysis pipeline')
    parser.add_argument('--output-dir', type=str, default=None,
                      help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--s3-output', type=str, default=None,
                      help='S3 output path (e.g., s3://bucket-name/path/to/output)')
    parser.add_argument('--sweep-id', type=str, default=None,
                      help='Process only runs from this sweep (default: all sweeps)')
    parser.add_argument('--run-id', type=str, default=None,
                      help='Process only this specific run')
    parser.add_argument('--device', type=str, default=None,
                      help=f'Device to use (default: {DEFAULT_DEVICE})')
    args = parser.parse_args()
    
    # Determine output directory (local or S3)
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
    
    # Set up logging with a directory based on OUTPUT_DIR
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger.info(f"Starting activation analysis pipeline. Log file: {log_file}")
    
    # Set device to use
    device = args.device or DEFAULT_DEVICE
    
    # Process each sweep
    sweeps_to_process = SWEEPS
    if args.sweep_id:
        if args.sweep_id in SWEEPS:
            sweeps_to_process = {args.sweep_id: SWEEPS[args.sweep_id]}
        else:
            logger.error(f"Sweep ID '{args.sweep_id}' not found in config.")
            return
    
    for sweep_id, model_type in sweeps_to_process.items():
        logger.info(f"Processing sweep {sweep_id} ({model_type})")
        
        # Initialize data manager
        model_data_manager = ModelDataManager(device=device)
        
        # Get runs to process
        if args.run_id:
            runs = [args.run_id]
            logger.info(f"Processing single run: {args.run_id}")
        else:
            # Get all runs in the sweep with 4 layers
            runs = [run_id for run_id in model_data_manager.list_runs_in_sweep(sweep_id) 
                    if 'L4' in run_id]
            logger.info(f"Found {len(runs)} runs with 4 layers in sweep {sweep_id}")
        
        # Process each run
        for run_id in runs:
            try:
                sweep_output_dir = output_dir
                if sweep_output_dir is None:
                    # Use default directory structure
                    sweep_output_dir = os.path.join(OUTPUT_DIR, sweep_id)
                    
                # For S3 paths, we might want to include the sweep ID in the path
                elif is_s3_path(sweep_output_dir) and not args.sweep_id:
                    # Add sweep ID to the path to maintain the same structure as local directories
                    sweep_output_dir = os.path.join(sweep_output_dir, sweep_id)
                    
                result = process_run(sweep_id, run_id, model_type, device=device, output_dir=sweep_output_dir)
                logger.info(result)
            except Exception as e:
                logger.error(f"Error processing run {run_id} in sweep {sweep_id}:")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()