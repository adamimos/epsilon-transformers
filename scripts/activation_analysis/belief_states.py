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
    
    def markov_approx_msps(self, run_config, max_order=3):
        """Run Markov approximations with caching."""
        # Try to load cached data
        cached_data = self.data_manager.load_markov_data(run_config['process_config'], max_order)
        if cached_data is not None:
            return cached_data
        
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
        
        # Save the computed data
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
            # Get Markov approximations data
            markov_data = self.markov_approx_msps(run_config, max_order=max_order)
            
            # Process each Markov order
            for order, data in enumerate(markov_data, 1):
                try:
                    # Unpack the data
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
                except Exception as e:
                    logger.error(f"Error processing Markov order {order}: {e}")
                    logger.error(f"Full error: {traceback.format_exc()}")
                    # Skip this order rather than using placeholder
                    logger.warning(f"Skipping Markov order {order} due to error")
            
            # Check if we have any valid classical beliefs
            if not classical_beliefs:
                raise ValueError("No valid Markov orders could be processed")
                
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