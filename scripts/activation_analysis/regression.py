"""
Regression analysis functions for mapping activations to belief states.
"""
import torch
import numpy as np
import pandas as pd
import logging
import traceback
from tqdm.auto import tqdm

from scripts.activation_analysis.utils import standardize, unstandardize_coefficients, report_variance_explained, setup_logging
from scripts.activation_analysis.config import REPORT_VARIANCE, DO_BASELINE

# Module logger
logger = logging.getLogger("regression")

class RegressionAnalyzer:
    """Class for performing regression analysis on activations."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def process_single_layer(self, act_tensor, belief_states, nn_word_probs, rcond, layer_name=None):
        """Process regression for a single layer."""
        try:
            # Check shapes and log information
            logger.debug(f"Processing layer with activation shape: {act_tensor.shape}")
            logger.debug(f"Belief states shape: {belief_states.shape}")
            logger.debug(f"Word probs shape: {nn_word_probs.shape}")
            
            # Reshape tensors for regression
            X = act_tensor.view(-1, act_tensor.shape[-1]).to(self.device)
            Y = belief_states.view(-1, belief_states.shape[-1]).to(self.device)
            weights = nn_word_probs.view(-1).to(self.device)
            
            # Check for shape compatibility
            if X.shape[0] != Y.shape[0] or X.shape[0] != weights.shape[0]:
                raise ValueError(f"Shape mismatch: X: {X.shape}, Y: {Y.shape}, weights: {weights.shape}")
                
            # Normalize weights
            weights = weights / weights.sum()
            
            # --- STEP 1: Standardize features ---
            X_std, mean, std = standardize(X)
            
            # --- STEP 2: SVD for variance analysis (separate from regression) ---
            if REPORT_VARIANCE:
                var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
            else:
                var_expl_str, singular_values = "", None
            
            # --- STEP 3: Weighted regression ---
            # Add bias term
            ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
            X_std_bias = torch.cat([ones, X_std], dim=1)
            
            # Apply weights
            sqrt_weights = torch.sqrt(weights).unsqueeze(1)
            X_weighted = X_std_bias * sqrt_weights
            Y_weighted = Y * sqrt_weights

            # Calculate beta using pseudoinverse with specified regularization
            A = X_weighted.T @ X_weighted
            beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)

            # Convert coefficients back to original scale
            beta = unstandardize_coefficients(beta_std, mean, std)
            
            # --- STEP 4: Evaluate regression ---
            # Make predictions
            ones_orig = torch.ones(X.shape[0], 1, device=X.device)
            X_orig_bias = torch.cat([ones_orig, X], dim=1)
            Y_pred = X_orig_bias @ beta
            
            # Calculate weighted error
            distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
            weighted_distances = distances * weights
            mean_dist = weighted_distances.sum()
            
            # Calculate R-squared
            total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
            explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
            r_squared = (explained_var / total_var).item()
            
            dims = X.shape[1]
            return mean_dist.item(), dims, var_expl_str, singular_values, beta.cpu().numpy(), r_squared
            
        except Exception as e:
            logger.error(f"Error in process_single_layer: {e}")
            raise
    
    def process_activation_layers(self, acts, belief_states, nn_word_probs, rcond, cached_svds=None):
        """
        Process regression for all activation layers and their combinations.
        
        Args:
            acts: Dictionary of activations by layer
            belief_states: Target belief states
            nn_word_probs: Probability weights
            rcond: Regularization parameter
            cached_svds: Optional dict of pre-computed SVDs to avoid redundant computation
            
        Returns:
            DataFrame of results, singular value dict, weights dict, best layer name, best layers dict
        """
        records = []
        best_layers = {}
        weights = {}
        singular_values_dict = {}
        best_idx = 0
        
        # Process each layer individually
        for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
            # Use cached SVD data if available
            if cached_svds and layer_name in cached_svds:
                X = cached_svds[layer_name]['X']
                X_std = cached_svds[layer_name]['X_std']
                Y = cached_svds[layer_name]['Y'] 
                weights_tensor = cached_svds[layer_name]['weights']
                singular_vals = cached_svds[layer_name]['singular_values']
                var_expl = cached_svds[layer_name]['var_expl']
                mean = cached_svds[layer_name]['mean']
                std = cached_svds[layer_name]['std']
                dims = X.shape[1]
                
                # Only need to compute the regression with the given rcond
                # Apply weights
                ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                X_std_bias = torch.cat([ones, X_std], dim=1)
                sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                X_weighted = X_std_bias * sqrt_weights
                Y_weighted = Y * sqrt_weights
                
                # Calculate beta using pseudoinverse with specified regularization
                A = X_weighted.T @ X_weighted
                beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                
                # Convert coefficients back to original scale
                beta = unstandardize_coefficients(beta_std, mean, std)
                
                # Make predictions and calculate metrics
                ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                X_orig_bias = torch.cat([ones_orig, X], dim=1)
                Y_pred = X_orig_bias @ beta
                
                # Calculate weighted error
                distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                weighted_distances = distances * weights_tensor
                norm_dist = weighted_distances.sum().item()
                
                # Calculate R-squared
                total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                r_squared = (explained_var / total_var).item()
                
            else:
                # Compute everything from scratch
                norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                    act_tensor, belief_states, nn_word_probs, rcond
                )
            
            # Store the weights for EVERY layer, not just the best one
            weights[layer_name] = beta
            
            records.append({
                "layer_name": layer_name,
                "layer_idx": layer_idx,
                "norm_dist": norm_dist,
                "dims": dims,
                "variance_explained": var_expl,
                "r_squared": r_squared
            })
            
            best_layers[layer_name] = norm_dist
            singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
            
            # Track best layer index
            if layer_idx == 0 or norm_dist < records[best_idx]["norm_dist"]:
                best_idx = len(records) - 1
        
        # Process concatenated activations - without input/embeddings
        non_input_layers = {}
        for k, v in acts.items():
            if not any(x in k.lower() for x in ['pre', 'embedding', 'input']):
                non_input_layers[k] = v
        logger.debug(f"Non-input layers: {list(non_input_layers.keys())}")
        
        if non_input_layers:
            layer_name = "concat_no_input"
            concat_acts = torch.cat([non_input_layers[k] for k in non_input_layers], dim=-1)
            
            # Use cached SVD if available
            if cached_svds and layer_name in cached_svds:
                X = cached_svds[layer_name]['X']
                X_std = cached_svds[layer_name]['X_std']
                Y = cached_svds[layer_name]['Y'] 
                weights_tensor = cached_svds[layer_name]['weights']
                singular_vals = cached_svds[layer_name]['singular_values']
                var_expl = cached_svds[layer_name]['var_expl']
                mean = cached_svds[layer_name]['mean']
                std = cached_svds[layer_name]['std']
                dims = X.shape[1]
                
                # Only need to compute the regression with the given rcond
                ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
                X_std_bias = torch.cat([ones, X_std], dim=1)
                sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
                X_weighted = X_std_bias * sqrt_weights
                Y_weighted = Y * sqrt_weights
                
                A = X_weighted.T @ X_weighted
                beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
                
                beta = unstandardize_coefficients(beta_std, mean, std)
                
                ones_orig = torch.ones(X.shape[0], 1, device=X.device)
                X_orig_bias = torch.cat([ones_orig, X], dim=1)
                Y_pred = X_orig_bias @ beta
                
                distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
                weighted_distances = distances * weights_tensor
                norm_dist = weighted_distances.sum().item()
                
                total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
                explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
                r_squared = (explained_var / total_var).item()
            else:
                norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                    concat_acts, belief_states, nn_word_probs, rcond
                )
            
            # Store weights for this layer too
            weights[layer_name] = beta
            
            records.append({
                "layer_name": layer_name,
                "layer_idx": layer_idx + 1,
                "norm_dist": norm_dist,
                "dims": dims,
                "variance_explained": var_expl,
                "r_squared": r_squared
            })
            
            best_layers[layer_name] = norm_dist
            singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
            
            if norm_dist < records[best_idx]["norm_dist"]:
                best_idx = len(records) - 1
        
        # Process all concatenated activations - with input/embeddings
        layer_name = "concat_all"
        concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
        
        # Use cached SVD if available
        if cached_svds and layer_name in cached_svds:
            X = cached_svds[layer_name]['X']
            X_std = cached_svds[layer_name]['X_std']
            Y = cached_svds[layer_name]['Y'] 
            weights_tensor = cached_svds[layer_name]['weights']
            singular_vals = cached_svds[layer_name]['singular_values']
            var_expl = cached_svds[layer_name]['var_expl']
            mean = cached_svds[layer_name]['mean']
            std = cached_svds[layer_name]['std']
            dims = X.shape[1]
            
            # Only need to compute the regression with the given rcond
            ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
            X_std_bias = torch.cat([ones, X_std], dim=1)
            sqrt_weights = torch.sqrt(weights_tensor).unsqueeze(1)
            X_weighted = X_std_bias * sqrt_weights
            Y_weighted = Y * sqrt_weights
            
            A = X_weighted.T @ X_weighted
            beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)
            
            beta = unstandardize_coefficients(beta_std, mean, std)
            
            ones_orig = torch.ones(X.shape[0], 1, device=X.device)
            X_orig_bias = torch.cat([ones_orig, X], dim=1)
            Y_pred = X_orig_bias @ beta
            
            distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
            weighted_distances = distances * weights_tensor
            norm_dist = weighted_distances.sum().item()
            
            total_var = torch.sum((Y_weighted - Y_weighted.mean(dim=0))**2)
            explained_var = torch.sum((Y_pred * sqrt_weights - Y_weighted.mean(dim=0))**2)
            r_squared = (explained_var / total_var).item()
        else:
            norm_dist, dims, var_expl, singular_vals, beta, r_squared = self.process_single_layer(
                concat_acts, belief_states, nn_word_probs, rcond
            )
        
        # Store weights for this layer too
        weights[layer_name] = beta
        
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx + 2 if 'layer_idx' in locals() else 1,
            "norm_dist": norm_dist,
            "dims": dims,
            "variance_explained": var_expl,
            "r_squared": r_squared
        })
        
        best_layers[layer_name] = norm_dist
        singular_values_dict[layer_name] = singular_vals.tolist() if hasattr(singular_vals, 'tolist') else None
        
        if norm_dist < records[best_idx]["norm_dist"]:
            best_idx = len(records) - 1
        
        best_layer = records[best_idx]["layer_name"]
        return pd.DataFrame(records), singular_values_dict, weights, best_layer, best_layers

    def analyze_checkpoint_with_activations(self, target_activations, belief_targets, rcond_sweep_list):
        """
        Analyze activations across all targets and regularization parameters.
        
        Args:
            target_activations: Dictionary mapping target names to their corresponding activation dictionaries
            belief_targets: Dictionary of belief targets to analyze
            rcond_sweep_list: List of regularization parameters to sweep
            
        Returns:
            all_results: List of DataFrames with results
            best_weights_dict: Dictionary of best weights
            best_singular_dict: Dictionary of singular values
        """
        all_results = []
        best_weights_dict = {}
        best_singular_dict = {}
        
        for target_name, target_data in belief_targets.items():
            if target_name not in target_activations:
                logger.warning(f"No activations found for target {target_name}, skipping")
                continue
                
            target = target_data['beliefs']
            probs = target_data['probs']
            acts = target_activations[target_name]
            
            target_best_weights = {}
            target_best_singular = {}
            
            # Log shapes for debugging
            logger.debug(f"Target {target_name} - Beliefs shape: {target.shape}, Probs shape: {probs.shape}")
            for layer_name, layer_acts in acts.items():
                logger.debug(f"  {layer_name} activation shape: {layer_acts.shape}")
            
            # Precompute SVD and other expensive operations for each layer
            cached_svds = {}
            logger.info("Precomputing SVD for each layer...")
            
            for layer_name, act_tensor in acts.items():
                try:
                    # Reshape tensors for regression
                    X = act_tensor.view(-1, act_tensor.shape[-1]).to(self.device)
                    Y = target.view(-1, target.shape[-1]).to(self.device)
                    weights_tensor = probs.view(-1).to(self.device)
                    weights_tensor = weights_tensor / weights_tensor.sum()
                    
                    # Standardize features
                    X_std, mean, std = standardize(X)
                    
                    # SVD for variance analysis
                    var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                    
                    # Cache computations
                    cached_svds[layer_name] = {
                        'X': X,
                        'X_std': X_std,
                        'Y': Y,
                        'weights': weights_tensor,
                        'mean': mean,
                        'std': std,
                        'singular_values': singular_values,
                        'var_expl': var_expl_str
                    }
                    
                    # Store singular values for later use - they don't depend on rcond
                    if layer_name not in target_best_singular:
                        target_best_singular[layer_name] = []
                    
                    if singular_values is not None:
                        target_best_singular[layer_name].append({
                            "singular_values": singular_values.tolist() if hasattr(singular_values, 'tolist') else singular_values
                        })
                except Exception as e:
                    logger.warning(f"Error precomputing SVD for layer {layer_name}: {e}")
                    # We'll fall back to non-cached computation for this layer
            
            # Now do the same for concatenated layers
            # Process concatenated activations - without input/embeddings
            non_input_layers = {}
            for k, v in acts.items():
                if not any(x in k.lower() for x in ['pre', 'embedding', 'input']):
                    non_input_layers[k] = v
                    
            if non_input_layers:
                try:
                    concat_acts = torch.cat([non_input_layers[k] for k in non_input_layers], dim=-1)
                    # Precompute SVD for concat_no_input
                    X = concat_acts.view(-1, concat_acts.shape[-1]).to(self.device)
                    Y = target.view(-1, target.shape[-1]).to(self.device)
                    weights_tensor = probs.view(-1).to(self.device)
                    weights_tensor = weights_tensor / weights_tensor.sum()
                    
                    X_std, mean, std = standardize(X)
                    var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                    
                    cached_svds["concat_no_input"] = {
                        'X': X,
                        'X_std': X_std,
                        'Y': Y,
                        'weights': weights_tensor,
                        'mean': mean,
                        'std': std,
                        'singular_values': singular_values,
                        'var_expl': var_expl_str
                    }
                except Exception as e:
                    logger.warning(f"Error precomputing SVD for concat_no_input: {e}")
            
            # Similarly for concat_all
            try:
                concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
                # Precompute SVD for concat_all
                X = concat_acts.view(-1, concat_acts.shape[-1]).to(self.device)
                Y = target.view(-1, target.shape[-1]).to(self.device)
                weights_tensor = probs.view(-1).to(self.device)
                weights_tensor = weights_tensor / weights_tensor.sum()
                
                X_std, mean, std = standardize(X)
                var_expl_str, singular_values = report_variance_explained(X_std, REPORT_VARIANCE)
                
                cached_svds["concat_all"] = {
                    'X': X,
                    'X_std': X_std,
                    'Y': Y,
                    'weights': weights_tensor,
                    'mean': mean,
                    'std': std,
                    'singular_values': singular_values,
                    'var_expl': var_expl_str
                }
            except Exception as e:
                logger.warning(f"Error precomputing SVD for concat_all: {e}")
            
            # Sweep across regularization parameters
            for rcond_val in tqdm(rcond_sweep_list, desc=f"rcond sweep for target {target_name}", leave=False):
                df, _, weights, best_layer, best_layers = self.process_activation_layers(
                    acts, target, probs, rcond_val, cached_svds=cached_svds
                )
                
                df['target'] = target_name
                df['rcond'] = rcond_val
                all_results.append(df)
                
                # Track best weights
                for layer, dist in best_layers.items():
                    if layer not in target_best_weights or dist < target_best_weights[layer]["dist"]:
                        target_best_weights[layer] = {
                            "weights": weights.get(layer, None),
                            "rcond": rcond_val,
                            "dist": dist
                        }
            
            best_weights_dict[target_name] = target_best_weights
            
            # Ensure we have singular values for this target
            if not target_best_singular:
                logger.warning(f"No singular values found for target {target_name}")
                # Create empty placeholder to prevent errors
                target_best_singular = {layer: [] for layer in acts.keys()}
            
            # Log how many singular value entries we're storing
            total_entries = sum(len(entries) for entries in target_best_singular.values())
            logger.info(f"Storing {total_entries} singular value entries for target {target_name}")
            
            best_singular_dict[target_name] = target_best_singular
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
            
        return combined_df, best_weights_dict, best_singular_dict

    def run_random_baseline(self, random_acts, belief_targets, rcond_val):
        """Run regression on random baseline activations."""
        if not DO_BASELINE:
            return [], {}, {}
            
        all_results = []
        best_singular_dict = {}
        best_weights_dict = {}
        
        for random_idx, acts in enumerate(tqdm(random_acts, desc="Random baselines")):
            for target_name, target_data in belief_targets.items():
                target = target_data['beliefs']
                probs = target_data['probs']
                
                df, singular_values, weights, _, best_layers = self.process_activation_layers(
                    acts, target, probs, rcond_val
                )
                
                df['checkpoint'] = f'RANDOM_{random_idx}'
                df['target'] = target_name
                df['rcond'] = rcond_val
                all_results.append(df)
                
                # Store singular values for first few random models
                if random_idx < 10:
                    if target_name not in best_singular_dict:
                        best_singular_dict[target_name] = {}
                    
                    for layer, sv in singular_values.items():
                        if layer not in best_singular_dict[target_name]:
                            best_singular_dict[target_name][layer] = []
                        
                        best_singular_dict[target_name][layer].append({
                            "random_idx": random_idx,
                            "singular_values": sv
                        })
                    
                    # Store best weights for this random model
                    if target_name not in best_weights_dict:
                        best_weights_dict[target_name] = {}
                    
                    for layer, dist in best_layers.items():
                        layer_key = f"{layer}_random_{random_idx}"
                        
                        if layer_key not in best_weights_dict[target_name]:
                            best_weights_dict[target_name][layer_key] = {
                                "weights": weights.get(layer, None),
                                "rcond": rcond_val,
                                "dist": dist,
                                "random_idx": random_idx
                            }
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
            
        return combined_df, best_singular_dict, best_weights_dict
    
    def run_random_baseline_streaming(self, random_act_generator, belief_targets, rcond_val, all_results=None, best_singular_dict=None, best_weights_dict=None):
        """Process a single random baseline at a time.
        
        This is a memory-efficient version that processes one random model at a time
        and appends results to the provided collections or creates new ones.
        
        Args:
            random_act_generator: Generator yielding (random_idx, activations) tuples
            belief_targets: Dictionary of belief targets
            rcond_val: Regularization parameter
            all_results: Optional list to accumulate DataFrames (created if None)
            best_singular_dict: Optional dict for singular values (created if None)
            best_weights_dict: Optional dict for weights (created if None)
            
        Returns:
            tuple: (all_results, best_singular_dict, best_weights_dict)
        """
        if not DO_BASELINE:
            return [], {}, {}
            
        # Initialize accumulators if not provided
        all_results = all_results or []
        best_singular_dict = best_singular_dict or {}
        best_weights_dict = best_weights_dict or {}
        
        # Process the current random network
        random_idx, acts = next(random_act_generator, (None, None))
        if random_idx is None:
            return all_results, best_singular_dict, best_weights_dict
            
        for target_name, target_data in belief_targets.items():
            target = target_data['beliefs']
            probs = target_data['probs']
            
            df, singular_values, weights, _, best_layers = self.process_activation_layers(
                acts, target, probs, rcond_val
            )
            
            df['checkpoint'] = f'RANDOM_{random_idx}'
            df['target'] = target_name
            df['rcond'] = rcond_val
            all_results.append(df)
            
            # Store singular values for first few random models
            if random_idx < 10:
                if target_name not in best_singular_dict:
                    best_singular_dict[target_name] = {}
                
                for layer, sv in singular_values.items():
                    if layer not in best_singular_dict[target_name]:
                        best_singular_dict[target_name][layer] = []
                    
                    best_singular_dict[target_name][layer].append({
                        "random_idx": random_idx,
                        "singular_values": sv
                    })
                
                # Store best weights for this random model
                if target_name not in best_weights_dict:
                    best_weights_dict[target_name] = {}
                
                for layer, dist in best_layers.items():
                    layer_key = f"{layer}_random_{random_idx}"
                    
                    if layer_key not in best_weights_dict[target_name]:
                        best_weights_dict[target_name][layer_key] = {
                            "weights": weights.get(layer, None),
                            "rcond": rcond_val,
                            "dist": dist,
                            "random_idx": random_idx
                        }
        
        return all_results, best_singular_dict, best_weights_dict