"""
Functions for generating and working with belief states.
"""
import torch
import logging
import traceback
import json
import os
from tqdm.auto import tqdm

from epsilon_transformers.process.GHMM import markov_approximation
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_nn_inputs

from scripts.activation_analysis.utils import setup_logging

# Module logger
logger = logging.getLogger("belief_states")

class BeliefStateGenerator:
    """Class for generating and managing belief states."""
    
    def __init__(self, data_manager, device='cpu'):
        self.data_manager = data_manager
        self.device = device
    
    def check_belief_dimensions(self, data_tuple, order, context=""):
        """Check belief dimensions and log results. Return True if should stop."""
        # Each data tuple has (nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized)
        beliefs = data_tuple[1]  # nn_beliefs is at index 1
        belief_dim = beliefs.shape[-1]
        logger.info(f"{context} Markov order {order}: belief dimension = {belief_dim}")
        
        if belief_dim >= 64:
            logger.info(f"{context} Belief dimension ({belief_dim}) at order {order} has reached or exceeded 64.")
            logger.info(f"{context} Stopping at Markov order {order}.")
            return True
        return False
    
    def markov_approx_msps(self, run_config, max_order=3):
        """Run Markov approximations with caching."""
        # Try to load cached data
        cached_data = self.data_manager.load_markov_data(run_config['process_config'], max_order)
        if cached_data is not None:
            # Check for early stopping in cached data
            stop_at_order = max_order
            for order, data_tuple in enumerate(cached_data, 1):
                if self.check_belief_dimensions(data_tuple, order, context="Cached"):
                    stop_at_order = order
                    break
            return cached_data[:stop_at_order]
        
        logger.info("Computing Markov approximation data...")
        T = get_matrix_from_args(**run_config['process_config'])
        ghmm = TransitionMatrixGHMM(T)
        markov_data = []
        
        # Calculate shared process details
        n_ctx = run_config['model_config']['n_ctx']
        
        for order in tqdm(range(1, max_order+1), desc="Running Markov Approximations"):
            markov_approx = markov_approximation(ghmm, order)
            msp = markov_approx.derive_mixed_state_tree(depth=n_ctx)
            data = self.prepare_data_from_msp(msp, n_ctx)
            markov_data.append(data)
            
            # Check if we should stop due to belief dimension
            if self.check_belief_dimensions(data, order, context="Computed"):
                break
        
        # Save the computed data (only what we actually computed)
        self.data_manager.save_markov_data(markov_data, run_config['process_config'], max_order)
        
        return markov_data
    
    def prepare_data_from_msp(self, msp, path_length):
        """
        Prepare data from a mixed state presentation tree.
        This is a simplified version adapted from activation_analysis.py
        """
        tree_paths = msp.paths
        tree_beliefs = msp.belief_states
        tree_unnormalized_beliefs = msp.unnorm_belief_states
        path_probs = msp.path_probs
        msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in tree_beliefs]
        msp_belief_index = {tuple(b): i for i, b in enumerate(set(msp_beliefs))}
        
        nn_paths = [x for x in tree_paths if len(x) == path_length]
        nn_inputs = torch.tensor(nn_paths, dtype=torch.int).clone().detach().to("cpu")

        probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}
        
        nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = get_beliefs_for_nn_inputs(
            nn_inputs,
            msp_belief_index,
            tree_paths,
            tree_beliefs,
            tree_unnormalized_beliefs,
            probs_dict
        )
        
        return nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs
    
    def generate_classical_belief_states(self, run_config, max_order=3):
        """
        Generate classical belief states (Markov Order approximations) for a given run configuration.
        
        Args:
            run_config: Run configuration containing process configuration
            max_order: Maximum Markov order to compute
            
        Returns:
            Dictionary mapping Markov order keys to corresponding belief state tensors
            or None if generation fails completely
        """
        logger.info(f"Generating classical belief states up to order {max_order}...")
        
        # Get model input shape for reference
        nn_inputs, nn_beliefs, _, nn_word_probs, _ = self.data_manager.load_msp_data(run_config)
        
        # Create a dictionary to hold the classical beliefs
        classical_beliefs = {}
        
        try:
            # Get Markov approximations data - this now includes early stopping logic
            markov_data = self.markov_approx_msps(run_config, max_order=max_order)
            
            # Log summary of computed Markov orders
            logger.info(f"Generated {len(markov_data)} Markov orders (may be less than max_order={max_order} due to dimension limit)")
            
            # Process each Markov order
            for order, data in enumerate(markov_data, 1):
                try:
                    # Original unpacking code - keep as is to minimize changes
                    m_inputs, beliefs, _, m_probs, _ = data
                    
                    logger.info(f"Markov order {order}:")
                    logger.info(f"  m_inputs.shape: {m_inputs.shape}")
                    logger.info(f"  beliefs.shape: {beliefs.shape}")
                    logger.info(f"  m_probs.shape: {m_probs.shape}")
                    
                    # Move to the specified device (don't reshape)
                    beliefs = beliefs.to(self.device)
                    m_inputs = m_inputs.to(self.device)
                    
                    # Add both inputs and beliefs to dictionary with proper key
                    classical_beliefs[f"markov_order_{order}"] = {
                        'inputs': m_inputs,
                        'beliefs': beliefs,
                        'probs': m_probs.to(self.device)
                    }
                    
                    logger.info(f"Markov order {order} beliefs shape: {beliefs.shape}")
                    
                    # We no longer need this check here since markov_approx_msps already implements early stopping
                    # But we'll keep it for redundancy and clarity in the logs
                    belief_dim = beliefs.shape[-1]
                    if belief_dim >= 64:
                        logger.info(f"Belief dimension ({belief_dim}) has reached or exceeded 64. Stopping at Markov order {order}.")
                        break
                except Exception as e:
                    logger.error(f"Error processing Markov order {order}: {e}")
                    logger.error(f"Full error: {traceback.format_exc()}")
                    # Skip this order rather than using placeholder
                    logger.warning(f"Skipping Markov order {order} due to error")
            
            # Check if we have any valid classical beliefs
            if not classical_beliefs:
                raise ValueError("No valid Markov orders could be processed")
            
            # Print a summary of all belief dimensions
            logger.info("=" * 50)
            logger.info("SUMMARY OF BELIEF DIMENSIONS:")
            for mk in sorted(classical_beliefs.keys()):
                belief_dim = classical_beliefs[mk]['beliefs'].shape[-1]
                logger.info(f"  {mk}: dimension = {belief_dim}")
            logger.info("=" * 50)
                
            return classical_beliefs
            
        except Exception as e:
            logger.error(f"Error generating classical belief states: {e}")
            logger.error(f"Full error: {traceback.format_exc()}")
            
            # Determine if we should continue or abort
            if not classical_beliefs:
                logger.critical("Failed to generate ANY classical belief states")
                return None
            else:
                logger.warning(f"Generated partial classical beliefs for {len(classical_beliefs)} orders")
                return classical_beliefs