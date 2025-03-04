# analysis_all_runs_with_unnorm_immediate_save.py

"""
Analysis script for neural network runs that works for both transformers and RNNs.

For each run in a given sweep:
  1. Load MSP data (inputs, normalized beliefs, unnormalized beliefs).
  2. Run a random–init control experiment (for normalized and unnormalized beliefs),
     using ALL seeds.
  3. For each checkpoint:
       a) Compute the MSE per layer (and on all–layer concatenated activations)
          for the trained model.
       b) Also compute MSE for a “shuffled–weights” control.
  4. Combine all data into a run–specific DataFrame and immediately save it.
  5. (Optionally) save an interactive figure.
  
This version supports transformer and RNN models. It uses a generic helper
for “random init” experiments and uses the provided get_activations() function that
branches on nn_type.
  
Requirements:
  - transformer_lens
  - Plotly
  - S3 access via S3ModelLoader
  - epsilon_transformers codebase and helper functions (included below)
"""

#%% [BLOCK 1]: Imports & Common Functions
import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import random
import plotly.graph_objects as go
import plotly.offline as pyo
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig

# Local modules
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import (
    prepare_msp_data,
    run_activation_to_beliefs_regression_fast,
    run_activation_to_beliefs_regression_fast_weighted,
    model_type,   # returns "transformer" or "rnn"
    get_activations,
    get_sweep_type
)

# --- A helper to extract a checkpoint number from its file path
def extract_checkpoint_number(ckpt_path: str) -> int:
    base = os.path.basename(ckpt_path)
    num_str, _ = os.path.splitext(base)
    try:
        return int(num_str)
    except ValueError:
        return num_str

# --- Compute layer–wise MSEs (including “all layers” concat)
def get_layer_mses_for_activations(acts: torch.Tensor, ground_truth_beliefs: torch.Tensor) -> list[float]:
    n_layers = acts.shape[0]
    mses = []
    for i in range(n_layers):
        mse_i = run_activation_to_beliefs_regression_fast(acts[i], ground_truth_beliefs)
        mses.append(float(mse_i))
    # Concatenated across layers:
    all_layers_acts = acts.permute(1, 2, 0, 3).reshape(acts.shape[1], acts.shape[2], -1)
    mse_concat = run_activation_to_beliefs_regression_fast(all_layers_acts, ground_truth_beliefs)
    mses.append(float(mse_concat))
    return mses

# --- Generic random-init control experiment function
def random_init_control_experiment_generic(
    model_config,        # For transformers: a HookedTransformerConfig; for RNNs: a dict with model settings
    nn_inputs: torch.Tensor,
    nn_beliefs: torch.Tensor,
    n_random_inits: int,
    nn_type: str,        # "transformer" or "rnn"
) -> pd.DataFrame:
    device = nn_inputs.device
    
    # Define a model constructor lambda based on nn_type.
    if nn_type == "transformer":
        # For transformers, assume model_config is a HookedTransformerConfig
        model_constructor = lambda config: HookedTransformer(config)
    elif nn_type == "rnn":
        # Here, model_config is the inner config.
        vocab_size = model_config.get("vocab_size", 100)  # Use an appropriate default if not present.
        from epsilon_transformers.training.networks import create_RNN  # adjust the import path as needed
        # Wrap the inner model_config inside another dictionary with key "model_config"
        model_constructor = lambda config: create_RNN({"model_config": config}, vocab_size, device)
    else:
        raise ValueError(f"Unsupported nn_type: {nn_type}")
    
    # Create a dummy model using the selected constructor.
    dummy_config = copy.deepcopy(model_config)
    dummy_model = model_constructor(dummy_config)
    
    with torch.no_grad():
        dummy_acts = get_activations(dummy_model, nn_inputs, nn_type)
    n_layers = dummy_acts.shape[0]
    
    records = []
    for seed in tqdm(range(n_random_inits), desc="Random seeds"):
        # Create a copy of the configuration and set the seed.
        config_copy = copy.deepcopy(model_config)
        if nn_type == "transformer":
            config_copy.seed = seed
        elif nn_type == "rnn":
            config_copy['seed'] = seed
        
        # Instantiate the model using the appropriate constructor.
        random_model = model_constructor(config_copy)
        with torch.no_grad():
            r_acts = get_activations(random_model, nn_inputs, nn_type)
        mses_this_seed = get_layer_mses_for_activations(r_acts, nn_beliefs)
        for layer_idx, mse_val in enumerate(mses_this_seed):
            records.append({
                "seed": seed,
                "layer_index": layer_idx,
                "MSE": mse_val,
                "random_or_trained": "random"
            })
    df = pd.DataFrame(records)
    return df



# --- Shuffle model weights (same for both transformer and RNN models)
def shuffle_model_weights(model) -> torch.nn.Module:
    shuffled_model = copy.deepcopy(model)
    for name, param in shuffled_model.named_parameters():
        if param.requires_grad and param.ndim > 1:
            flat = param.data.view(-1)
            perm = torch.randperm(flat.size(0))
            shuffled_flat = flat[perm]
            param.data.copy_(shuffled_flat.view(param.data.size()))
    return shuffled_model

# --- Plotting figure for a single run (using Plotly)
def plot_run_figure(
    df_run: pd.DataFrame,
    run_id: str,
    outpath_html: str
):
    fig = go.Figure()
    # Group by norm_type, random_or_trained, and layer_index
    groups = df_run.groupby(["norm_type", "random_or_trained", "layer_index"])
    for (norm_type, randtrain, layer_idx), df_grp in groups:
        if randtrain == "random":
            xvals = df_grp["seed"]
            xaxis_name = "seed"
        else:
            xvals = df_grp["checkpoint"]
            xaxis_name = "checkpoint"
        fig.add_trace(go.Scatter(
            x=xvals,
            y=df_grp["MSE"],
            mode="markers+lines",
            name=f"{randtrain}-{norm_type}-L{layer_idx}",
            hovertemplate=(
                f"Layer={layer_idx}<br>"
                f"Norm={norm_type}<br>"
                f"Type={randtrain}<br>"
                f"MSE=%{{y:.5f}}<extra></extra>"
            )
        ))
    fig.update_layout(
        title=f"MSE vs. Checkpoint/Seed (Run={run_id})",
        xaxis_title="Checkpoint (if trained/control) / Seed (if random)",
        yaxis_title="MSE",
        yaxis_type="log",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    pyo.plot(fig, filename=outpath_html, auto_open=False)
    print(f"Saved figure to {outpath_html}")

#%% [BLOCK 2]: Main Analysis Over All Runs (Immediate Save)

# Create a loader for S3 (assumed to be properly configured)
loader = S3ModelLoader()

# Here we choose the sweep. (For transformer runs you might have e.g. "20241205175736";
# for RNN runs, use the folder name "20241121152808".)
sweeps = {
    # Example: "sweep_id": "Model_Type"
    "20241205175736": "Transformer",
    "20241121152808": "RNN"
}
# For this run, choose the desired sweep_id (change as appropriate)
sweep_id = list(sweeps.keys())[0]   # e.g. using the second key for RNN runs

runs = loader.list_runs_in_sweep(sweep_id)
print(f"Sweep {sweep_id} has runs:", runs)

# Create a local cache folder for each run
base_outdir = f"analysis_cache_word_probs_pinv/{sweep_id}"
os.makedirs(base_outdir, exist_ok=True)

for run_id in runs:
    print(f"\n=== Analyzing run: {run_id} ===")
    run_outdir = os.path.join(base_outdir, run_id)
    os.makedirs(run_outdir, exist_ok=True)
    
    run_csv_path = os.path.join(run_outdir, "run_data.csv")
    if os.path.exists(run_csv_path):
        print(f"Already found {run_csv_path}, skipping re-analysis.")
        continue

    # 1) List checkpoints
    all_ckpts = loader.list_checkpoints(sweep_id, run_id)
    if not all_ckpts:
        print("No checkpoints found; skipping.")
        continue

    # 2) Load a sample checkpoint to obtain model configuration and MSP data
    sample_ckpt = all_ckpts[0]
    device = "cpu"
    model, run_config = loader.load_checkpoint(sweep_id, run_id, sample_ckpt, device=device)
    
    # Determine model type ("transformer" or "rnn")
    nn_type = model_type(model)
    print(f"Model type for run {run_id}: {nn_type}")
    
    # (If you wish to restrict to one type, you might check here—but now we support both.)
    # For example, remove any check that would skip non–transformer runs.

    # Prepare MSP data.
    # (Assumes your run_config contains both a "process_config" and "model_config" key.)
    nn_inputs, nn_beliefs, _, _, nn_unnormalized_beliefs = prepare_msp_data(
        run_config, run_config["model_config"]
    )
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_unnormalized_beliefs = nn_unnormalized_beliefs.to(device)

    # 3) Run random–init control experiment (normalized beliefs)
    n_seeds = 10
    transformer_config = run_config["model_config"]
    print(f"Random–init control (normalized) for run={run_id} ...")
    df_random_norm = random_init_control_experiment_generic(
        transformer_config, nn_inputs, nn_beliefs, n_seeds, nn_type
    )
    df_random_norm["norm_type"] = "normalized"
    df_random_norm["checkpoint"] = "RANDOM"
    df_random_norm["sweep_id"] = sweep_id
    df_random_norm["run_id"] = run_id

    # 4) Run random–init control experiment (unnormalized beliefs)
    print(f"Random–init control (unnormalized) for run={run_id} ...")
    df_random_unnorm = random_init_control_experiment_generic(
        transformer_config, nn_inputs, nn_unnormalized_beliefs, n_seeds, nn_type
    )
    df_random_unnorm["norm_type"] = "unnormalized"
    df_random_unnorm["checkpoint"] = "RANDOM"
    df_random_unnorm["sweep_id"] = sweep_id
    df_random_unnorm["run_id"] = run_id

    # 5) Checkpoint analysis
    print(f"Analyzing {len(all_ckpts)} checkpoints for run={run_id} ...")
    records_ckpt = []
    for ckpt in tqdm(all_ckpts, desc="Checkpoints"):
        model_ckpt, _ = loader.load_checkpoint(sweep_id, run_id, ckpt, device=device)
        with torch.no_grad():
            # Use the generic get_activations that will branch based on nn_type.
            acts_ckpt = get_activations(model_ckpt, nn_inputs, nn_type)
        mses_norm = get_layer_mses_for_activations(acts_ckpt, nn_beliefs)
        mses_unnorm = get_layer_mses_for_activations(acts_ckpt, nn_unnormalized_beliefs)
        n_layers = acts_ckpt.shape[0]
        for layer_idx in range(n_layers+1):
            records_ckpt.append({
                "sweep_id": sweep_id,
                "run_id": run_id,
                "checkpoint": extract_checkpoint_number(ckpt),
                "norm_type": "normalized",
                "random_or_trained": "trained",
                "seed": -1,
                "layer_index": layer_idx,
                "MSE": mses_norm[layer_idx]
            })
            records_ckpt.append({
                "sweep_id": sweep_id,
                "run_id": run_id,
                "checkpoint": extract_checkpoint_number(ckpt),
                "norm_type": "unnormalized",
                "random_or_trained": "trained",
                "seed": -1,
                "layer_index": layer_idx,
                "MSE": mses_unnorm[layer_idx]
            })
        # --- Shuffled weights control ---
        model_ckpt_shuffled = shuffle_model_weights(model_ckpt)
        with torch.no_grad():
            acts_ckpt_shuffled = get_activations(model_ckpt_shuffled, nn_inputs, nn_type)
        mses_norm_shuffled = get_layer_mses_for_activations(acts_ckpt_shuffled, nn_beliefs)
        mses_unnorm_shuffled = get_layer_mses_for_activations(acts_ckpt_shuffled, nn_unnormalized_beliefs)
        for layer_idx in range(n_layers+1):
            records_ckpt.append({
                "sweep_id": sweep_id,
                "run_id": run_id,
                "checkpoint": extract_checkpoint_number(ckpt),
                "norm_type": "normalized",
                "random_or_trained": "trained_shuffled",
                "seed": -1,
                "layer_index": layer_idx,
                "MSE": mses_norm_shuffled[layer_idx]
            })
            records_ckpt.append({
                "sweep_id": sweep_id,
                "run_id": run_id,
                "checkpoint": extract_checkpoint_number(ckpt),
                "norm_type": "unnormalized",
                "random_or_trained": "trained_shuffled",
                "seed": -1,
                "layer_index": layer_idx,
                "MSE": mses_unnorm_shuffled[layer_idx]
            })

    df_ckpt = pd.DataFrame(records_ckpt)

    # 6) Combine all DataFrames into one run–specific DataFrame.
    df_run = pd.concat([df_random_norm, df_random_unnorm, df_ckpt], ignore_index=True)
    print(f"Shape of final DF for run {run_id}: {df_run.shape}")

    # 7) Save run CSV
    df_run.to_csv(run_csv_path, index=False)
    print(f"Saved run data to {run_csv_path}")

    # 8) Save figure (optional – uncomment if desired)
    # figure_outpath = os.path.join(run_outdir, "mse_plot.html")
    # plot_run_figure(df_run, run_id, figure_outpath)
    print(f"Analysis done for run {run_id} -- inspect the CSV (and figure if saved).")

print("=== Done analyzing all runs. ===")
# %%
