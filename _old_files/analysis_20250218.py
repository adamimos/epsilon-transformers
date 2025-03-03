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

#%% Initialize
loader = S3ModelLoader()
sweep_id = '20241205175736' # transformers
max_layers = 10
relevant_activation_keys = ['blocks.0.hook_resid_pre'] + [f'blocks.{i}.hook_resid_post' for i in range(max_layers)] + ['ln_final.hook_normalized']
base_outdir = f"analysis_cache_20250220_pinv/{sweep_id}"
os.makedirs(base_outdir, exist_ok=True)

def run_linear_regression_with_probs_old(X, Y, nn_word_probs):

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

def run_linear_regression_with_probs(X, Y, nn_word_probs):

    # Add bias term (column of ones)
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Create weight matrix W
    W = np.diag(nn_word_probs)
    
    # Weighted linear regression using pseudoinverse: beta = (X'WX)^(-1)X'WY
    XtW = X_with_bias.T @ W
    beta = np.linalg.pinv(XtW @ X_with_bias) @ XtW @ Y
    
    # Get predictions
    Y_pred = X_with_bias @ beta
    
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
print(f"runs are {loader.list_runs_in_sweep(sweep_id)}")
start_ind = 0
for run_id in loader.list_runs_in_sweep(sweep_id)[start_ind:]:

    if os.path.exists(os.path.join(base_outdir, f"{run_id}.csv")):
        print(f"Skipping run {run_id} because it already exists.")
        continue
    else:
        print(f"Processing run {run_id}...")

    # if run_id == 'run_7_L1_H4_DH16_DM64_mess3': then run otherwise dont
    #if run_id != 'run_7_L1_H4_DH16_DM64_mess3':
    #    continue

    print(f"Processing run {run_id}...")

    results = []
    init_checkpoint = loader.list_checkpoints(sweep_id, run_id)[0]
    model, run_config = loader.load_checkpoint(sweep_id, run_id, init_checkpoint, device='cpu')
    nn_inputs, nn_beliefs, _, nn_word_probs, nn_unnormalized_beliefs = prepare_msp_data(run_config, run_config["model_config"]
    )

    model_cfg = model.cfg
    for rndm_seed in tqdm(range(10), desc="Random seeds"):
        model_cfg.seed = rndm_seed
        random_model = HookedTransformer(model_cfg)
        _, acts = random_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = 'RANDOM'
        df['random_or_trained'] = 'random'
        results.append(df)

    for checkpoint in tqdm(loader.list_checkpoints(sweep_id, run_id), desc="Trained checkpoints"):
        checkpoint_number = extract_checkpoint_number(checkpoint)
        model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint, device='cpu')
        _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)

        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'trained'
        print(df)
        results.append(df)

        # Shuffle weights
        shuffled_model = shuffle_weights(model)
        _, acts = shuffled_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
        df = process_ckpt(acts, nn_beliefs, nn_unnormalized_beliefs, nn_word_probs)
        df['checkpoint'] = checkpoint_number
        df['random_or_trained'] = 'shuffled'
        results.append(df)

    results_df = pd.concat(results)
    results_df.to_csv(os.path.join(base_outdir, f"{run_id}_pinv.csv"), index=False)
    print(results_df)


# %%

