"""
Script to compare the original and optimized pseudoinverse implementations.

This script runs both the original and optimized pseudoinverse implementations
on the same data and compares the results to verify they are equivalent.
"""

import os
import torch
import logging
import argparse
import pandas as pd
from tqdm.auto import tqdm

from scripts.activation_analysis.config import *
from scripts.activation_analysis.utils import setup_logging
from scripts.activation_analysis.data_loading import ModelDataManager
from scripts.activation_analysis.belief_states import BeliefStateGenerator
from scripts.activation_analysis.regression import RegressionAnalyzer

# Set up module logger
logger = logging.getLogger("compare_pinv")

def compare_implementations(sweep_id, run_id, model_type, device='cpu'):
    """
    Compare original and optimized pseudoinverse implementations on a specific run.
    
    Args:
        sweep_id: Sweep ID to use
        run_id: Run ID to process
        model_type: Model type (transformer or mlp)
        device: Device to use (cpu or cuda)
        
    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing pseudoinverse implementations on {sweep_id}/{run_id}")
    
    # Initialize components
    model_data_manager = ModelDataManager(device=device)
    belief_generator = BeliefStateGenerator(model_data_manager, device=device)
    
    # Create both types of regression analyzers
    regression_analyzer_original = RegressionAnalyzer(device=device, use_efficient_pinv=False)
    regression_analyzer_optimized = RegressionAnalyzer(device=device, use_efficient_pinv=True)
    
    # Create a regression analyzer with comparison mode
    regression_analyzer_compare = RegressionAnalyzer(device=device)
    
    # Load an initial checkpoint to get configuration
    all_checkpoints = model_data_manager.list_checkpoints(sweep_id, run_id)
    if not all_checkpoints:
        logger.warning(f"No checkpoints found for run {run_id}")
        return None
        
    init_checkpoint = all_checkpoints[0]
    model, run_config = model_data_manager.load_checkpoint(sweep_id, run_id, init_checkpoint)
    model = model.to(device)
    
    # Load model inputs and network belief states
    logger.info("Loading mixed state presentation data")
    nn_inputs, nn_beliefs, _, nn_word_probs, _ = model_data_manager.load_msp_data(run_config)
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_word_probs = nn_word_probs.to(device)
    
    # Generate classical belief states
    logger.info("Generating classical belief states...")
    classical_beliefs = belief_generator.generate_classical_belief_states(
        run_config, max_order=MAX_MARKOV_ORDER
    )
    
    # Extract activations for neural network beliefs
    logger.info("Extracting activations...")
    act_extractor = ActivationExtractor(device=device)
    relevant_keys = TRANSFORMER_ACTIVATION_KEYS if model_type == 'transformer' else None
    nn_acts = act_extractor.extract_activations(model, nn_inputs, model_type, relevant_keys)
    
    # Prepare belief targets dictionary and corresponding activations
    belief_targets = {}
    target_activations = {}
    
    # Add neural network beliefs
    belief_targets['nn_beliefs'] = {
        'beliefs': nn_beliefs,
        'probs': nn_word_probs
    }
    target_activations['nn_beliefs'] = nn_acts
    
    # Add classical beliefs if available
    if classical_beliefs:
        for mk in classical_beliefs:
            belief_targets[mk] = {
                'beliefs': classical_beliefs[mk]['beliefs'],
                'probs': classical_beliefs[mk]['probs']
            }
            # Extract activations using the markov-specific inputs
            mk_inputs = classical_beliefs[mk]['inputs']
            mk_acts = act_extractor.extract_activations(model, mk_inputs, model_type, relevant_keys)
            target_activations[mk] = mk_acts
    
    # Define a subset of rcond values to test (for faster testing)
    rcond_values = RCOND_SWEEP_LIST  # Use just the first two rcond values for quicker testing
    
    # --- METHOD 1: Run both implementations separately and compare afterwards ---
    logger.info("METHOD 1: Running implementations separately")
    
    # Process using original implementation
    logger.info("Running original implementation...")
    start_time = time.time()
    results_original, _, _ = regression_analyzer_original.analyze_checkpoint_with_activations(
        target_activations, belief_targets, rcond_values
    )
    original_time = time.time() - start_time
    logger.info(f"Original implementation took {original_time:.2f} seconds")
    
    # Process using optimized implementation
    logger.info("Running optimized implementation...")
    start_time = time.time()
    results_optimized, _, _ = regression_analyzer_optimized.analyze_checkpoint_with_activations(
        target_activations, belief_targets, rcond_values
    )
    optimized_time = time.time() - start_time
    logger.info(f"Optimized implementation took {optimized_time:.2f} seconds")
    
    # Compare the numerical results
    results_merged = pd.merge(
        results_original, 
        results_optimized,
        on=['layer_name', 'layer_idx', 'target', 'rcond'],
        suffixes=('_orig', '_opt')
    )
    
    # Calculate differences
    results_merged['norm_dist_diff'] = results_merged['norm_dist_orig'] - results_merged['norm_dist_opt']
    results_merged['norm_dist_rel_diff'] = (results_merged['norm_dist_diff'] / results_merged['norm_dist_orig']).abs()
    results_merged['r_squared_diff'] = results_merged['r_squared_orig'] - results_merged['r_squared_opt']
    
    # Show summary of differences
    logger.info(f"Average absolute relative difference in norm_dist: {results_merged['norm_dist_rel_diff'].mean():.8f}")
    logger.info(f"Max absolute relative difference in norm_dist: {results_merged['norm_dist_rel_diff'].max():.8f}")
    logger.info(f"Average absolute difference in r_squared: {results_merged['r_squared_diff'].abs().mean():.8f}")
    logger.info(f"Speedup factor: {original_time / optimized_time:.2f}x")
    
    # --- METHOD 2: Use the built-in comparison mode ---
    logger.info("\nMETHOD 2: Using built-in comparison mode")
    logger.info("Running built-in comparison...")
    start_time = time.time()
    results_compare, _, _ = regression_analyzer_compare.analyze_checkpoint_with_activations(
        target_activations, belief_targets, rcond_values, compare_implementations=True
    )
    compare_time = time.time() - start_time
    logger.info(f"Comparison mode took {compare_time:.2f} seconds")
    
    return results_merged

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compare pseudoinverse implementations')
    parser.add_argument('--sweep-id', type=str, default=None, required=True,
                      help='Sweep ID to process')
    parser.add_argument('--run-id', type=str, default=None, required=True,
                      help='Run ID to process')
    parser.add_argument('--device', type=str, default=None,
                      help=f'Device to use (default: {DEFAULT_DEVICE})')
    parser.add_argument('--output-csv', type=str, default=None,
                      help='Path to save comparison results CSV')
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    log_file = setup_logging(log_dir=log_dir)
    logger = logging.getLogger("compare_pinv")
    logger.info(f"Starting pseudoinverse implementation comparison. Log file: {log_file}")
    
    # Set device to use
    device = args.device or DEFAULT_DEVICE
    logger.info(f"Using device: {device}")
    
    if not args.sweep_id or not args.run_id:
        logger.error("Both sweep-id and run-id are required.")
        return
    
    # Get model type from config
    if args.sweep_id in SWEEPS:
        model_type = SWEEPS[args.sweep_id]
    else:
        logger.warning(f"Unknown sweep ID: {args.sweep_id}. Assuming transformer model.")
        model_type = 'transformer'
    
    # Run the comparison
    results = compare_implementations(args.sweep_id, args.run_id, model_type, device)
    
    # Save results if requested
    if args.output_csv and results is not None:
        results.to_csv(args.output_csv, index=False)
        logger.info(f"Saved comparison results to {args.output_csv}")

if __name__ == "__main__":
    import time
    from scripts.activation_analysis.data_loading import ActivationExtractor
    main() 