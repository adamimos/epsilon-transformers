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

from config import (
    OUTPUT_DIR, DEFAULT_DEVICE, SWEEPS, RCOND_SWEEP_LIST, 
    MAX_MARKOV_ORDER, TRANSFORMER_ACTIVATION_KEYS,
    MAX_CHECKPOINTS, PROCESS_ALL_CHECKPOINTS
)
from data_loading import ModelDataManager, ActivationExtractor
from belief_states import BeliefStateGenerator
from regression import RegressionAnalyzer
from utils import (
    save_results_to_h5, extract_checkpoint_number, setup_logging,
    save_unified_results, save_results_csv, save_results_in_csv_format
)

# Set up module logger
logger = logging.getLogger("main")

def process_checkpoint(checkpoint, sweep_id, run_id, model_type, 
                      model_data_manager, act_extractor, belief_analyzer, 
                      classical_beliefs, nn_beliefs, nn_inputs, nn_word_probs,
                      random_acts_cache=None):
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

def process_run(sweep_id, run_id, model_type, device=DEFAULT_DEVICE):
    """Process a single run."""
    # Only process runs with 'L4' in run_id (4-layer networks)
    if 'L4' not in run_id:
        logger.info(f"Skipping run {run_id} (not a 4-layer network).")
        return f"Skipped run {run_id} (not a 4-layer network)."

    logger.info(f"Processing run {run_id} (model type: {model_type}) on device {device}")

    # Setup output directory
    base_outdir = os.path.join(OUTPUT_DIR, sweep_id)
    os.makedirs(base_outdir, exist_ok=True)

    # Initialize components
    model_data_manager = ModelDataManager(device=device)
    act_extractor = ActivationExtractor(device=device)
    belief_generator = BeliefStateGenerator(model_data_manager, device=device)
    regression_analyzer = RegressionAnalyzer(device=device)

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

    # Generate classical belief states (Markov approximations)
    logger.info("Generating classical belief states")
    classical_beliefs = belief_generator.generate_classical_belief_states(
        run_config, max_order=MAX_MARKOV_ORDER
    )

    # Generate random baseline activations if needed
    random_acts_cache = {}
    relevant_keys = TRANSFORMER_ACTIVATION_KEYS if model_type == 'transformer' else None
    
    # Compute random baselines for each target type
    logger.info("Generating random baseline activations")
    # For neural network beliefs
    random_acts_cache['nn_beliefs'] = act_extractor.get_random_activations(
        model, run_config, nn_inputs, model_type, relevant_keys
    )
    
    # For classical beliefs - use the proper inputs
    if classical_beliefs is not None:
        for mk in classical_beliefs:
            mk_inputs = classical_beliefs[mk]['inputs']
            random_acts_cache[mk] = act_extractor.get_random_activations(
                model, run_config, mk_inputs, model_type, relevant_keys
            )
    
    # Run random baseline analysis with full rcond sweep
    logger.info("Running random baseline analysis with rcond sweep")
    
    # Prepare belief targets for random baseline
    baseline_targets = {}
    if classical_beliefs is not None:
        for mk in classical_beliefs:
            baseline_targets[mk] = {
                'beliefs': classical_beliefs[mk]['beliefs'],
                'probs': classical_beliefs[mk]['probs']
            }
    baseline_targets['nn_beliefs'] = {
        'beliefs': nn_beliefs,
        'probs': nn_word_probs
    }
    
    # Run random baseline analysis for each target type with rcond sweep
    baseline_results = []
    baseline_singular_values = {}
    baseline_weights = {}  # Add a dictionary to collect weights from random networks
    
    for target_name, random_acts_list in random_acts_cache.items():
        if target_name not in baseline_targets:
            logger.warning(f"No belief target found for {target_name}, skipping random baseline")
            continue
            
        for random_idx, random_acts in enumerate(random_acts_list):
            if random_idx >= 10:  # Limit to first 10 random models for efficiency
                break
                
            # Sweep through all rcond values
            for rcond_val in tqdm(RCOND_SWEEP_LIST, desc=f"rcond sweep for random model {random_idx}, target {target_name}", leave=False):
                # Run regression for this random model with this rcond value
                df, singular_values, weights, _, best_layers = regression_analyzer.process_activation_layers(
                    random_acts, 
                    baseline_targets[target_name]['beliefs'],
                    baseline_targets[target_name]['probs'],
                    rcond_val
                )
                
                df['checkpoint'] = f'RANDOM_{random_idx}'
                df['target'] = target_name
                df['rcond'] = rcond_val
                baseline_results.append(df)
                
                # Store best weights for the lowest rcond value (most relevant)
                if rcond_val == min(RCOND_SWEEP_LIST):
                    if target_name not in baseline_weights:
                        baseline_weights[target_name] = {}
                        
                    for layer, dist in best_layers.items():
                        layer_key = f"{layer}_random_{random_idx}"
                        
                        if layer_key not in baseline_weights[target_name]:
                            baseline_weights[target_name][layer_key] = {
                                "weights": weights.get(layer, None),
                                "rcond": rcond_val,
                                "dist": dist,
                                "random_idx": random_idx
                            }
            
            # Store singular values just once per random model since they don't depend on rcond
            if target_name not in baseline_singular_values:
                baseline_singular_values[target_name] = {}
                
            for layer, sv in singular_values.items():
                if layer not in baseline_singular_values[target_name]:
                    baseline_singular_values[target_name][layer] = []
                
                baseline_singular_values[target_name][layer].append({
                    "random_idx": random_idx,
                    "singular_values": sv
                })
    
    # Process checkpoint
    if baseline_results:
        baseline_df = pd.concat(baseline_results, ignore_index=True)
        baseline_df['sweep_id'] = sweep_id
        baseline_df['run_id'] = run_id
        
        # Random baseline data for unified format
        random_data = {
            'df': baseline_df,
            'singular': baseline_singular_values,
            'weights': baseline_weights  # Add weights to random data
        }
        logger.info(f"Generated random baseline data for {len(random_acts_cache)} target types")
        logger.info(f"Random baseline analysis complete with {len(baseline_results)} results")
    else:
        random_data = None

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
                nn_word_probs,
                random_acts_cache
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
    # Set up logging with a directory based on OUTPUT_DIR
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger.info(f"Starting activation analysis pipeline. Log file: {log_file}")
    
    # Process each sweep
    for sweep_id, model_type in SWEEPS.items():
        logger.info(f"Processing sweep {sweep_id} ({model_type})")
        
        # Initialize data manager
        model_data_manager = ModelDataManager(device=DEFAULT_DEVICE)
        
        # Get all runs in the sweep with 4 layers
        runs = [run_id for run_id in model_data_manager.list_runs_in_sweep(sweep_id) 
                if 'L4' in run_id]
        
        logger.info(f"Found {len(runs)} runs with 4 layers in sweep {sweep_id}")
        
        # Process each run
        for run_id in runs:
            try:
                result = process_run(sweep_id, run_id, model_type)
                logger.info(result)
            except Exception as e:
                logger.error(f"Error processing run {run_id} in sweep {sweep_id}:")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()