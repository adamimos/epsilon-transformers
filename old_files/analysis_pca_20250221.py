#%% imports
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from tqdm.auto import tqdm
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
import copy
import os

#%% Device setup
# On macOS, if MPS is available, use it; otherwise use CUDA if available, else CPU.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#%% Initialize
loader = S3ModelLoader()
sweep_id = '20241205175736'  # transformers
max_layers = 10
relevant_activation_keys = (
    ['blocks.0.hook_resid_pre'] +
    [f'blocks.{i}.hook_resid_post' for i in range(max_layers)] +
    ['ln_final.hook_normalized']
)
base_outdir = f"analysis_cache_20250220_noPCA/{sweep_id}"
os.makedirs(base_outdir, exist_ok=True)

def run_linear_regression_with_probs(X, Y, nn_word_probs):
    """
    Weighted linear regression using the pseudoinverse.
    All operations are performed on GPU using torch.
    """
    # Add bias term (a column of ones)
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X_with_bias = torch.cat([ones, X], dim=1)
    
    # Multiply X_with_bias.T with weights using broadcasting
    XtW = X_with_bias.t() * nn_word_probs.unsqueeze(0)
    XtWX = XtW @ X_with_bias
    XtWY = XtW @ Y
    beta = torch.pinverse(XtWX) @ XtWY
    
    # Get predictions and compute weighted Euclidean distances
    Y_pred = X_with_bias @ beta
    distances = torch.sqrt(torch.sum((Y_pred - Y) ** 2, dim=1))
    weighted_distances = distances * nn_word_probs
    mean_dist = weighted_distances.sum()
    return mean_dist

def process_single_layer(act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    """
    Process activations for a single layer by performing weighted linear regression
    on the full activation dimensions (i.e., without any dimensionality reduction).
    
    Steps:
      1. Reshape activations to 2D.
      2. Move data to the GPU.
      3. Normalize sample weights.
      4. Run weighted linear regression on the original activations.
    """
    # Reshape activations to 2D and move to device
    X = act_tensor.view(-1, act_tensor.shape[-1]).to(device)
    
    # Get and normalize sample weights
    weights = nn_word_probs.view(-1).to(device)
    weights = weights / weights.sum()
    
    # Prepare target outputs on device
    Y_norm = nn_beliefs.view(-1, nn_beliefs.shape[-1]).to(device)
    Y_unnorm = nn_unnormalized_beliefs.view(-1, nn_unnormalized_beliefs.shape[-1]).to(device)
    
    # Run weighted linear regression on full activations
    norm_dist = run_linear_regression_with_probs(X, Y_norm, weights)
    unnorm_dist = run_linear_regression_with_probs(X, Y_unnorm, weights)
    
    # Return original feature dimension for reference
    return norm_dist.item(), unnorm_dist.item(), X.shape[1]

def process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs):
    """Process activations from a single checkpoint."""
    records = []
    for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
        norm_dist, unnorm_dist, original_dims = process_single_layer(
            act_tensor, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs
        )
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "norm_dist": norm_dist,
            "unnorm_dist": unnorm_dist,
            "original_dims": original_dims
        })

    # Process concatenated activations (across layers) in the same way
    concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
    norm_dist, unnorm_dist, original_dims = process_single_layer(
        concat_acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs
    )
    records.append({
        "layer_name": "concat",
        "layer_idx": layer_idx + 1,
        "norm_dist": norm_dist,
        "unnorm_dist": unnorm_dist,
        "original_dims": original_dims
    })
    return pd.DataFrame(records)

def shuffle_weights(model: HookedTransformer) -> HookedTransformer:
    """
    Returns a copy of the model with its weights shuffled.
    For each parameter tensor (with ndim > 1) that requires gradients,
    randomly permute its elements while preserving the original shape.
    """
    shuffled_model = copy.deepcopy(model)
    for name, param in shuffled_model.named_parameters():
        if param.requires_grad and param.ndim > 1:
            flat = param.data.view(-1)
            perm = torch.randperm(flat.size(0), device=param.device)
            shuffled_flat = flat[perm]
            param.data.copy_(shuffled_flat.view(param.data.size()))
    return shuffled_model

def extract_checkpoint_number(ckpt_path: str) -> int:
    """
    Extract the checkpoint number from a file path.
    """
    base = os.path.basename(ckpt_path)
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        return num_str

#%% Main Loop
num_runs = len(loader.list_runs_in_sweep(sweep_id))
print(f"runs are {loader.list_runs_in_sweep(sweep_id)}")
start_ind = 0
for run_id in loader.list_runs_in_sweep(sweep_id)[start_ind:]:
    if os.path.exists(os.path.join(base_outdir, f"{run_id}.csv")):
        print(f"Skipping run {run_id} because it already exists.")
        continue
    else:
        print(f"Processing run {run_id}...")

    results = []
    init_checkpoint = loader.list_checkpoints(sweep_id, run_id)[0]
    # Load the model to the chosen device
    model, run_config = loader.load_checkpoint(sweep_id, run_id, init_checkpoint, device=device)
    model = model.to(device)
    
    # Prepare data and move it to the device
    nn_inputs, nn_beliefs, _, nn_word_probs, nn_unnormalized_beliefs = prepare_msp_data(run_config, run_config["model_config"])
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_word_probs = nn_word_probs.to(device)
    nn_unnormalized_beliefs = nn_unnormalized_beliefs.to(device)

    model_cfg = model.cfg
    for rndm_seed in tqdm(range(10), desc="Random seeds"):
        model_cfg.seed = rndm_seed
        random_model = HookedTransformer(model_cfg).to(device)
        _, acts = random_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = 'RANDOM'
        df['random_or_trained'] = 'random'
        results.append(df)

    for checkpoint in tqdm(loader.list_checkpoints(sweep_id, run_id), desc="Trained checkpoints"):
        checkpoint_number = extract_checkpoint_number(checkpoint)
        model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint, device=device)
        model = model.to(device)
        _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)

        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'trained'
        print(df)
        results.append(df)

        # Process shuffled weights as control
        shuffled_model = shuffle_weights(model)
        _, acts = shuffled_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'shuffled'
        results.append(df)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(base_outdir, f"{run_id}_pinv_gpu.csv"), index=False)
    print(results_df)

# %%
