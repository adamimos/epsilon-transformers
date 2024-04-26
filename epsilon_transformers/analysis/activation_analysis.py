from typing import Iterator, List, Tuple, Dict
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer
from jaxtyping import Float
import numpy as np
import torch

from epsilon_transformers.process.MixedStateTree import MixedStateTree
from epsilon_transformers.process.Process import Process
from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.training.configs.training_configs import ProcessDatasetConfig

# TODO: TQDM find_msp_subpace_in_residual_stream
# TODO: (??) _generate_belief_state_and_activation() (??)
# TODO: Make dataclass for ground_truth & predicted_belief states
# TODO: Abstract persister & make persister for the dataclass

# TODO: All of this passing of the process around is unnecessary. I just gotta pass the msp around and that's it
# TODO: This train/test split is unnatural.... I may want to fix it or leave a None options
# TODO: For non toy models, derive_mixed_state_presentation needs to be parallelized
# TODO: Move _predicted_mixed_state_belief_vector to MSP class (??)
# TODO: Refactor this whole mess so that you only have to iterate through the samples list once foo
# TODO: Generalize for the case when we have to extract multiple different parts of the residual stream
# TODO: Add batch_size to generate_belief_state_and_activations


def get_beliefs_for_transformer_inputs(
    transformer_inputs: Float[torch.Tensor, "batch n_ctx"], 
    msp_belief_index: Dict[Tuple[float, ...], int], 
    tree_paths: List[List[int]], 
    tree_beliefs: List[List[float]]
) -> Tuple[Float[torch.Tensor, "batch n_ctx belief_dim"], Float[torch.Tensor, "batch n_ctx"]]:
    batch, n_ctx = transformer_inputs.shape
    belief_dim = len(list(msp_belief_index.keys())[0])
    path_belief_dict = {tuple(path): belief for path, belief in zip(tree_paths, tree_beliefs)}
    X_beliefs = torch.zeros(batch, n_ctx, belief_dim, device=transformer_inputs.device)
    X_belief_indices = torch.zeros(batch, n_ctx, dtype=torch.int, device=transformer_inputs.device)
    
    for i in range(batch):
        for j in range(n_ctx):
            input_substring = transformer_inputs[i, :j+1].cpu().numpy()
            belief_state = path_belief_dict[tuple(input_substring)]
            belief_state = np.round(belief_state, 5)
            X_beliefs[i, j] = torch.tensor(belief_state, dtype=torch.float32, device=transformer_inputs.device)
            X_belief_indices[i, j] = msp_belief_index[tuple(belief_state)]
    
    return X_beliefs, X_belief_indices

def generate_belief_state_and_activations(model: HookedTransformer, process: Process, num_sequences: int) -> Tuple[Float[np.ndarray, "num_samples n_ctx num_states"], Float[np.ndarray, "num_samples n_ctx d_model"]]:
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Create the Mixed State Presentation Tree
    msp_tree = process.derive_mixed_state_presentation(depth=model.cfg.n_ctx + 1) # For really large models this should definately be parallelized
    
    belief_states = []
    activations = []
    for x in process.yield_emission_histories(sequence_len=model.cfg.n_ctx, num_sequences=num_sequences):
        belief_states.append(msp_tree.path_to_beliefs(x))
        _, cache = model.run_with_cache(torch.tensor([x], device=device))
        activations.append(cache['ln_final.hook_normalized'].squeeze(0).cpu())
    
    # Turn into tensors
    belief_states = np.stack(belief_states)
    activations = np.array(activations)

    return belief_states, activations

def find_msp_subspace_in_residual_stream(model: HookedTransformer, process: Process, num_sequences: int) -> Tuple[Float[np.ndarray, "num_tokens num_states"], Float[np.ndarray, "num_tokens num_states"]]:
    ground_truth_belief_states, activations = generate_belief_state_and_activations(model=model, process=process, num_sequences=num_sequences)

    # Reshape activations
    ground_truth_belief_states_reshaped = ground_truth_belief_states.reshape(-1, ground_truth_belief_states.shape[-1])
    activations_reshaped = activations.reshape(-1, activations.shape[-1])
    
    # Run Linear Regression
    reg = LinearRegression().fit(activations_reshaped, ground_truth_belief_states_reshaped)
    predicted_beliefs = reg.predict(activations_reshaped)
    
    return ground_truth_belief_states_reshaped, predicted_beliefs

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained('pythia-160m')
    process = ZeroOneR()
    num_samples = 12

    find_msp_subspace_in_residual_stream(model = model, process=process, num_sequences=num_samples)