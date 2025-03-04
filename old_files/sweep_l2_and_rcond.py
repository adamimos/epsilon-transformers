
import os
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Common imports for loading data and models.
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import prepare_msp_data
from transformer_lens import HookedTransformer
from epsilon_transformers.training.networks import create_RNN

#%% Device setup (shared)
if torch.backends.mps.is_available():
    default_device = torch.device("mps")
elif torch.cuda.is_available():
    default_device = torch.device("cuda:0")
else:
    default_device = torch.device("cpu")
print("Default device:", default_device)

#%% Global Settings and Shared Helper Functions
USE_PCA = False
REPORT_VARIANCE = True

# Flag to control whether to perform the initial random baseline computations.
DO_BASELINE = False

# Number of random baseline models to compute.
NUM_RANDOM_BASELINES = 10

loader = S3ModelLoader()
max_layers = 10

# For transformers, these are the activation keys we are interested in.
def get_transformer_activation_keys():
    return (
        ['blocks.0.hook_resid_pre'] +
        [f'blocks.{i}.hook_resid_post' for i in range(max_layers)] +
        ['ln_final.hook_normalized']
    )

def standardize(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)
    X_std = (X - mean) / std
    return X_std, mean, std

def unstandardize_coefficients(beta_std, mean, std):
    beta0_std = beta_std[0:1, :]
    beta_rest_std = beta_std[1:, :]
    beta_rest = beta_rest_std / std.T
    beta0 = beta0_std - (mean / std).matmul(beta_rest_std)
    beta = torch.cat([beta0, beta_rest], dim=0)
    return beta

def compute_variance_explained(X_std):
    U, S, Vh = torch.linalg.svd(X_std, full_matrices=False)
    explained_variance = S**2
    total_variance = explained_variance.sum()
    explained_ratio = (explained_variance / total_variance).cpu().numpy()
    cumulative = np.cumsum(explained_ratio)
    return cumulative

def report_variance_explained(X_std):
    cumulative = compute_variance_explained(X_std)
    return str(cumulative.tolist())

# Updated weighted ridge regression function that always uses the pseudoinverse.
def weighted_linear_regression_ridge(X, Y, nn_word_probs, rcond=1e-5, l2_reg=0.0):
    """
    Perform weighted ridge regression using the pseudoinverse.
    
    We first standardize X (and add a bias term) then compute:
    
       beta = pinv(X_w^T X_w + l2_reg * I, rcond) @ (X_w^T Y_w),
    
    where X_w = sqrt(nn_word_probs) * [1, X] and Y_w = sqrt(nn_word_probs) * Y.
    """
    X_std, mean, std = standardize(X)
    var_expl_str = report_variance_explained(X_std) if REPORT_VARIANCE else ""
    ones = torch.ones(X_std.shape[0], 1, device=X_std.device)
    X_std_bias = torch.cat([ones, X_std], dim=1)
    sqrt_weights = torch.sqrt(nn_word_probs).unsqueeze(1)
    X_weighted = X_std_bias * sqrt_weights
    Y_weighted = Y * sqrt_weights

    I = torch.eye(X_weighted.shape[1], device=X_weighted.device)
    A = X_weighted.T @ X_weighted + l2_reg * I
    beta_std = torch.pinverse(A, rcond=rcond) @ (X_weighted.T @ Y_weighted)

    beta = unstandardize_coefficients(beta_std, mean, std)
    ones_orig = torch.ones(X.shape[0], 1, device=X.device)
    X_orig_bias = torch.cat([ones_orig, X], dim=1)
    Y_pred = X_orig_bias @ beta
    distances = torch.sqrt(torch.sum((Y_pred - Y)**2, dim=1))
    weighted_distances = distances * nn_word_probs
    mean_dist = weighted_distances.sum()
    return mean_dist, var_expl_str

# Modified regression on a single layer: we now only compute the norm fit.
def process_single_layer_ridge(act_tensor, nn_beliefs, nn_word_probs, device, rcond, l2_reg):
    X = act_tensor.view(-1, act_tensor.shape[-1]).to(device)
    Y = nn_beliefs.view(-1, nn_beliefs.shape[-1]).to(device)
    weights = nn_word_probs.view(-1).to(device)
    weights = weights / weights.sum()
    norm_dist, var_expl_str = weighted_linear_regression_ridge(X, Y, weights, rcond=rcond, l2_reg=l2_reg)
    dims = X.shape[1]
    return norm_dist.item(), dims, var_expl_str

def process_ckpt_ridge(acts, nn_beliefs, nn_word_probs, device, rcond, l2_reg):
    records = []
    for layer_idx, (layer_name, act_tensor) in enumerate(acts.items()):
        norm_dist, dims, var_expl = process_single_layer_ridge(
            act_tensor, nn_beliefs, nn_word_probs, device, rcond, l2_reg
        )
        records.append({
            "layer_name": layer_name,
            "layer_idx": layer_idx,
            "norm_dist": norm_dist,
            "dims": dims,
            "variance_explained": var_expl
        })
    # Also process the concatenated activations.
    concat_acts = torch.cat([acts[k] for k in acts], dim=-1)
    norm_dist, dims, var_expl = process_single_layer_ridge(
        concat_acts, nn_beliefs, nn_word_probs, device, rcond, l2_reg
    )
    records.append({
        "layer_name": "concat",
        "layer_idx": layer_idx + 1,
        "norm_dist": norm_dist,
        "dims": dims,
        "variance_explained": var_expl
    })
    return pd.DataFrame(records)

def extract_checkpoint_number(ckpt_path: str):
    base = os.path.basename(ckpt_path)
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        return num_str

#%% Model-Specific Activation Functions

def get_transformer_activations(model, nn_inputs, device, relevant_activation_keys):
    _, acts = model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
    return acts

def get_rnn_activations(model, nn_inputs, device):
    _, state_dict = model.forward_with_all_states(nn_inputs)
    acts = state_dict['layer_states']
    acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
    return acts_dict

def get_random_activations(model, run_config, nn_inputs, device, model_type, relevant_activation_keys=None):
    random_acts = []
    if model_type == 'transformer':
        model_cfg = copy.deepcopy(model.cfg)
        for rndm_seed in range(NUM_RANDOM_BASELINES):
            model_cfg.seed = rndm_seed
            random_model = HookedTransformer(model_cfg).to(device)
            _, acts = random_model.run_with_cache(nn_inputs, names_filter=lambda x: x in relevant_activation_keys)
            random_acts.append(acts)
    elif model_type == 'rnn':
        for rndm_seed in range(NUM_RANDOM_BASELINES):
            random_model = create_RNN(run_config, model.output_layer.out_features, device=device)
            _, state_dict = random_model.forward_with_all_states(nn_inputs)
            acts = state_dict['layer_states']
            acts_dict = {f"layer{i}": acts[i] for i in range(acts.shape[0])}
            random_acts.append(acts_dict)
    return random_acts

#%% Single Process Run Function (shared)
def process_run(sweep_id, run_id, model_type):
    # Only process runs that contain 'L4' in the run_id.
    if 'L4' not in run_id:
        return f"Skipping run {run_id} (does not contain 'L4')."
    
    device = default_device
    print(f"Processing run {run_id} (model type: {model_type}) on device {device}")
    
    base_outdir = os.path.join("analysis_cache_20250220_standardized_noPCA", sweep_id)
    os.makedirs(base_outdir, exist_ok=True)
    out_csv = os.path.join(base_outdir, f"{run_id}.csv")

    if os.path.exists(out_csv):
        return f"Skipping run {run_id} because it already exists."
    
    results = []
    # Load an initial checkpoint to obtain configuration.
    init_checkpoint = loader.list_checkpoints(sweep_id, run_id)[0]
    model, run_config = loader.load_checkpoint(sweep_id, run_id, init_checkpoint, device=device)
    model = model.to(device)
    nn_inputs, nn_beliefs, _, nn_word_probs, _ = prepare_msp_data(
        run_config, run_config["model_config"]
    )
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_word_probs = nn_word_probs.to(device)
    
    if model_type == 'transformer':
        relevant_activation_keys = get_transformer_activation_keys()
    
    # --- Random Baseline Computations (if enabled) ---
    if DO_BASELINE:
        random_acts_list = get_random_activations(
            model, run_config, nn_inputs, device, model_type,
            relevant_activation_keys if model_type=='transformer' else None
        )
        for acts in random_acts_list:
            # For the baseline we use default parameters: rcond=1e-15 and lambda=0.
            df = process_ckpt_ridge(acts, nn_beliefs, nn_word_probs, device,
                                      rcond=1e-15, l2_reg=0.0)
            df['checkpoint'] = 'RANDOM'
            df['sweep_type'] = 'baseline'
            df['lambda'] = 0.0
            df['rcond'] = 1e-15
            results.append(df)
    
    # --- Process Checkpoints (only every 20th checkpoint) ---
    all_checkpoints = loader.list_checkpoints(sweep_id, run_id)
    for idx, checkpoint in tqdm(enumerate(all_checkpoints)):
        if idx % 25 != 0:
            continue
        checkpoint_number = extract_checkpoint_number(checkpoint)
        model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint, device=device)
        model = model.to(device)
        if model_type == 'transformer':
            acts = get_transformer_activations(model, nn_inputs, device, relevant_activation_keys)
        else:
            acts = get_rnn_activations(model, nn_inputs, device)
        
        # --- L2 Regularization Sweep ---
        # Fixed rcond = 1e-15, lambda sweeps through: 0, 1e-10, 1e-9, 1e-8, ..., 1e-2.
        l2_sweep_list = [0.0] + list(np.logspace(-10, -2, num=9))
        for l2_val in l2_sweep_list:
            df = process_ckpt_ridge(acts, nn_beliefs, nn_word_probs, device,
                                    rcond=1e-15, l2_reg=l2_val)
            df['checkpoint'] = checkpoint_number
            df['sweep_type'] = 'l2_sweep'
            df['lambda'] = l2_val
            df['rcond'] = 1e-15
            results.append(df)
        
        # --- rcond Sweep ---
        # Fixed lambda = 0, sweep through different rcond values.
        rcond_sweep_list = [1e-15, 1e-10, 1e-5, 1e-2]
        for rcond_val in rcond_sweep_list:
            df = process_ckpt_ridge(acts, nn_beliefs, nn_word_probs, device,
                                    rcond=rcond_val, l2_reg=0.0)
            df['checkpoint'] = checkpoint_number
            df['sweep_type'] = 'rcond_sweep'
            df['lambda'] = 0.0
            df['rcond'] = rcond_val
            results.append(df)
    
    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        results_df = pd.DataFrame()
    
    
    results_df.to_csv(out_csv, index=False)
    loss_df = loader.load_loss_from_run(run_id=run_id, sweep_id=sweep_id)
    loss_csv = os.path.join(base_outdir, f"{run_id}_loss.csv")
    loss_df.to_csv(loss_csv, index=False)
    
    return f"Processed run {run_id} (model: {model_type}) on device {device}. Regression & sweep results saved to {out_csv}, loss saved to {loss_csv}"

#%% Main Loop over Sweeps

if __name__ == "__main__":
    # Define the sweeps and their model types.
    sweeps = {
        '20241205175736': 'transformer',
        '20241121152808': 'rnn'
    }
    
    for sweep_id, model_type in sweeps.items():
        runs = loader.list_runs_in_sweep(sweep_id)
        # Only process runs whose run_id contains 'L4'
        runs = [run_id for run_id in runs if 'L4' in run_id]
        print(f"Found {len(runs)} runs for sweep {sweep_id} ({model_type}) containing 'L4'.")
        for run_id in runs:
            try:
                result = process_run(sweep_id, run_id, model_type)
                print(result)
            except Exception as e:
                print(f"Error processing run {run_id} in sweep {sweep_id}: {e}")
