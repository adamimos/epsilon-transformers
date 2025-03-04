#%% imports
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from tqdm.auto import tqdm
import torch
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
import copy
import os
from epsilon_transformers.training.networks import create_RNN

#%% Initialize
loader = S3ModelLoader()
sweep_id = '20241121152808' # RNNs
base_outdir = f"analysis_cache_20250218/{sweep_id}"
os.makedirs(base_outdir, exist_ok=True)

def run_linear_regression_with_probs(X, Y, nn_word_probs):

    # Fit linear regression with sample weights
    reg = LinearRegression()
    reg.fit(X, Y, sample_weight=nn_word_probs)

    # Get predictions
    Y_pred = reg.predict(X)
    
    # Calculate Euclidean distances
    distances = np.sqrt(np.sum((Y_pred - Y)**2, axis=1))
    
    # Weight distances by word probabilities
    weighted_distances = distances * nn_word_probs
    mean_dist = weighted_distances.sum()
    
    return mean_dist

def process_single_layer(act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    """Run linear regression for a single layer's activations."""
    norm_dist = run_linear_regression_with_probs(
        act_tensor.numpy().reshape(-1, act_tensor.shape[-1]),
        nn_beliefs.numpy().reshape(-1, nn_beliefs.shape[-1]),
        nn_word_probs.numpy().reshape(-1)/nn_word_probs.numpy().reshape(-1).sum()
    )
    unnorm_dist = run_linear_regression_with_probs(
        act_tensor.numpy().reshape(-1, act_tensor.shape[-1]),
        nn_unnormalized_beliefs.numpy().reshape(-1, nn_unnormalized_beliefs.shape[-1]),
        nn_word_probs.numpy().reshape(-1)/nn_word_probs.numpy().reshape(-1).sum()
    )
    return norm_dist, unnorm_dist

def process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    """Process a single checkpoint."""
    records = []
    for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
        norm_dist, unnorm_dist = process_single_layer(act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "norm_dist": norm_dist,
            "unnorm_dist": unnorm_dist
        })

    concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
    norm_dist, unnorm_dist = process_single_layer(concat_acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
    records.append({
        "layer_name": "concat",
        "layer_idx": layer_idx+1,
        "norm_dist": norm_dist,
        "unnorm_dist": unnorm_dist
    })
    return pd.DataFrame(records)

def shuffle_weights(model: HookedTransformer) -> HookedTransformer:
    """
    Returns a copy of the model with its weights shuffled.
    For each parameter tensor that requires gradients (i.e. weights),
    we randomly permute its elements while preserving the original shape.
    (This is a simple control to break the learned structure while keeping
    the distribution of weight values.)
    """
    shuffled_model = copy.deepcopy(model)
    for name, param in shuffled_model.named_parameters():
        # If the parameter is a weight (typically requires grad) and is not a bias:
        if param.requires_grad and param.ndim > 1:
            flat = param.data.view(-1)
            perm = torch.randperm(flat.size(0))
            shuffled_flat = flat[perm]
            param.data.copy_(shuffled_flat.view(param.data.size()))
    return shuffled_model

def extract_checkpoint_number(ckpt_path: str) -> int:
    """
    Given a full checkpoint file path, extract the integer checkpoint number.
    Example:
      Input: "20241205175736/run_0_L1_H4_DH16_DM64_post_quantum/409804800.pt"
      Output: 409804800
    """
    # Get the file name (e.g. "409804800.pt")
    base = os.path.basename(ckpt_path)
    # Remove the file extension (".pt") to get just the number as string
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        # If conversion fails, return as is (or handle as appropriate)
        return num_str
#%% Main Loop
num_runs = len(loader.list_runs_in_sweep(sweep_id))
start_ind = 0
for run_id in loader.list_runs_in_sweep(sweep_id)[start_ind:]:

    if os.path.exists(os.path.join(base_outdir, f"{run_id}.csv")):
        print(f"Skipping run {run_id} because it already exists.")
        continue
    else:
        print(f"Processing run {run_id}...")

    results = []
    init_checkpoint = loader.list_checkpoints(sweep_id, run_id)[0]
    model, run_config = loader.load_checkpoint(sweep_id, run_id, init_checkpoint, device='cpu')
    nn_inputs, nn_beliefs, _, nn_word_probs, nn_unnormalized_beliefs = prepare_msp_data(run_config, run_config["model_config"]
    )

    rnn_cfg = run_config["model_config"]
    for rndm_seed in tqdm(range(10), desc="Random seeds"):
        random_model = create_RNN(run_config, model.output_layer.out_features, device='cpu')
        a, b = random_model.forward_with_all_states(nn_inputs)
        acts = b['layer_states']
        acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
        df = process_ckpt(acts_dict, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = 'RANDOM'
        df['random_or_trained'] = 'random'
        results.append(df)

    for checkpoint in tqdm(loader.list_checkpoints(sweep_id, run_id), desc="Trained checkpoints"):
        checkpoint_number = extract_checkpoint_number(checkpoint)
        model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint, device='cpu')
        a, b = model.forward_with_all_states(nn_inputs)
        acts = b['layer_states']
        acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
        df = process_ckpt(acts_dict, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'trained'
        results.append(df)

        # Shuffle weights
        shuffled_model = shuffle_weights(model)
        a, b = shuffled_model.forward_with_all_states(nn_inputs)
        acts = b['layer_states']
        acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
        df = process_ckpt(acts_dict, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'shuffled'
        results.append(df)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(base_outdir, f"{run_id}.csv"), index=False)
    print(results_df)


# %%

