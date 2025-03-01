"""
Functions for loading models, checkpoints, and activations.
"""
import torch
import io
import numpy as np
import logging
import copy
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from epsilon_transformers.training.networks import create_RNN

from scripts.activation_analysis.config import NUM_RANDOM_BASELINES, TRANSFORMER_ACTIVATION_KEYS
from scripts.activation_analysis.utils import setup_logging

# Module logger
logger = logging.getLogger("data_loading")

class ActivationExtractor:
    """Class for extracting activations from different model types."""
    
    def __init__(self, device):
        self.device = device
    
    def get_transformer_activations(self, model, nn_inputs, relevant_activation_keys):
        """Extract activations from a transformer model."""
        _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        return acts
    
    def get_rnn_activations(self, model, nn_inputs):
        """Extract RNN activations and add one-hot encoded inputs."""
        # Get RNN activations
        _, state_dict = model.forward_with_all_states(nn_inputs)
        acts = state_dict['layer_states']
        acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
        
        # Determine vocabulary size from output_layer
        if hasattr(model, 'output_layer'):
            vocab_size = model.output_layer.out_features
        else:
            raise AttributeError("Model does not have an 'output_layer' attribute to determine vocabulary size.")
        
        # Create one-hot encoded input
        one_hot_inputs = torch.zeros(nn_inputs.shape[0], nn_inputs.shape[1], vocab_size, device=self.device)
        
        # Convert indices to int64 (long) to fix the scatter_ error
        indices = nn_inputs.long().unsqueeze(-1)
        one_hot_inputs.scatter_(2, indices, 1)
        
        acts_dict["input"] = one_hot_inputs
        
        return acts_dict
    
    def get_random_activations(self, model, run_config, nn_inputs, model_type, relevant_activation_keys=None):
        """Generate activations from randomly initialized models for baseline comparison."""
        random_acts = []
        
        if model_type == 'transformer':
            model_cfg = copy.deepcopy(model.cfg)
            for rndm_seed in tqdm(range(NUM_RANDOM_BASELINES), desc="Generating random transformer networks"):
                model_cfg.seed = rndm_seed
                random_model = HookedTransformer(model_cfg).to(self.device)
                random_acts.append(self.get_transformer_activations(
                    random_model, nn_inputs, relevant_activation_keys
                ))
                
        elif model_type == 'rnn':
            for rndm_seed in tqdm(range(NUM_RANDOM_BASELINES), desc="Generating random RNN networks"):
                random_model = create_RNN(run_config, model.output_layer.out_features, device=self.device)
                random_acts.append(self.get_rnn_activations(random_model, nn_inputs))
                
        return random_acts

    def extract_activations(self, model, nn_inputs, model_type, relevant_activation_keys=None):
        """Extract activations based on model type."""
        if model_type == 'transformer':
            return self.get_transformer_activations(model, nn_inputs, relevant_activation_keys)
        else:
            return self.get_rnn_activations(model, nn_inputs)

class ModelDataManager:
    """Class for managing model data loading and caching."""
    
    def __init__(self, loader=None, device='cpu'):
        self.loader = loader or S3ModelLoader()
        self.device = device
        
    def load_checkpoint(self, sweep_id, run_id, checkpoint, device=None):
        """Load model checkpoint."""
        device = device or self.device
        return self.loader.load_checkpoint(sweep_id, run_id, checkpoint, device=device)
    
    def list_checkpoints(self, sweep_id, run_id, limit=None):
        """List available checkpoints for a run."""
        checkpoints = self.loader.list_checkpoints(sweep_id, run_id)
        if limit:
            return checkpoints[:limit]
        return checkpoints
    
    def list_runs_in_sweep(self, sweep_id, filter_func=None):
        """List available runs in a sweep, with optional filtering."""
        runs = self.loader.list_runs_in_sweep(sweep_id)
        if filter_func:
            runs = [run_id for run_id in runs if filter_func(run_id)]
        return runs
    
    def load_loss_from_run(self, sweep_id, run_id):
        """Load loss dataframe for a run."""
        return self.loader.load_loss_from_run(run_id=run_id, sweep_id=sweep_id)
    
    def load_msp_data(self, run_config):
        """Load mixed state presentation data for a run configuration."""
        nn_inputs, nn_beliefs, nn_indices, nn_word_probs, nn_unnormalized = prepare_msp_data(
            run_config, run_config["model_config"]
        )
        return nn_inputs, nn_beliefs, nn_indices, nn_word_probs, nn_unnormalized

    def save_markov_data(self, markov_data, process_config, max_order):
        """Save Markov approximation data to S3."""
        from scripts.activation_analysis.utils import get_markov_cache_key
        
        cache_key = get_markov_cache_key(process_config, max_order)
        path = f"analysis/markov_data/{cache_key}"
        
        # Process and save each Markov order separately
        for order, data in enumerate(markov_data, 1):
            nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized = data
            
            # Convert tensors to numpy arrays
            serializable_data = {
                'nn_inputs': nn_inputs.cpu().numpy(),
                'nn_beliefs': nn_beliefs.cpu().numpy(),
                'nn_indices': nn_indices.cpu().numpy(),
                'nn_probs': nn_probs.cpu().numpy(),
                'nn_unnormalized': nn_unnormalized.cpu().numpy(),
                'process_config': process_config,
                'order': order,
                'max_order': max_order
            }
            
            # Save with numpy's compressed format
            buf = io.BytesIO()
            np.savez_compressed(buf, **serializable_data)
            buf.seek(0)
            
            self.loader.s3_client.put_object(
                Bucket=self.loader.bucket_name,
                Key=f"{path}/order_{order}.npz",
                Body=buf.getvalue()
            )
        
        logger.info(f"Saved Markov data for orders 1-{max_order} to S3 cache")

    def load_markov_data(self, process_config, max_order):
        """Load Markov approximation data from S3 if it exists."""
        from scripts.activation_analysis.utils import get_markov_cache_key
        
        cache_key = get_markov_cache_key(process_config, max_order)
        path = f"analysis/markov_data/{cache_key}"
        
        try:
            # Check if data exists for all orders
            markov_data = []
            for order in range(1, max_order + 1):
                response = self.loader.s3_client.get_object(
                    Bucket=self.loader.bucket_name,
                    Key=f"{path}/order_{order}.npz"
                )
                
                # Load the compressed numpy data
                with io.BytesIO(response['Body'].read()) as buf:
                    data = np.load(buf, allow_pickle=True)
                    
                    # Convert back to tensors
                    nn_inputs = torch.from_numpy(data['nn_inputs'])
                    nn_beliefs = torch.from_numpy(data['nn_beliefs'])
                    nn_indices = torch.from_numpy(data['nn_indices'])
                    nn_probs = torch.from_numpy(data['nn_probs'])
                    nn_unnormalized = torch.from_numpy(data['nn_unnormalized'])
                    
                    markov_data.append((nn_inputs, nn_beliefs, nn_indices, nn_probs, nn_unnormalized))
            
            logger.info("Loaded cached Markov approximation data")
            return markov_data
                
        except Exception as e:
            logger.warning(f"Error loading cached Markov data: {e}")
            return None

def get_markov_beliefs(config, order, cache=None):
    """Load Markov beliefs."""
    from scripts.activation_analysis.utils import get_markov_cache_key

def generate_markov_beliefs(config, order, cache=None):
    """Generate Markov beliefs from scratch."""
    from scripts.activation_analysis.utils import get_markov_cache_key