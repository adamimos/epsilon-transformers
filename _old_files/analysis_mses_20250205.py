# analysis_all_runs_with_unnorm_immediate_save.py

"""
analysis_all_runs_with_unnorm_immediate_save.py

Script Outline:

1. We pick a sweep ID and iterate over *all runs* in that sweep, one by one.
2. For each run:
   a) Load MSP data (nn_inputs, normalized beliefs, unnormalized beliefs).
   b) Run random-init control for (normalized) and (unnormalized) beliefs, storing ALL seeds.
   c) For each checkpoint, compute MSE for each layer (and concatenated layers) for:
         - the trained model,
         - a new control: the trained model with shuffled weights.
   d) Combine random, trained, and trained_shuffled data into a single DataFrame with columns:
       [
         'sweep_id', 'run_id', 'checkpoint', 'norm_type',
         'random_or_trained', 'seed', 'layer_index', 'MSE'
       ]
   e) Save this run-specific DataFrame as a CSV right away.
   f) Save a run-specific figure (HTML or PNG) showing MSE across checkpoints per layer,
      plus random-init "bands" or distribution for reference.

This way, each run's analysis is *independently cached*. You can inspect partial results without waiting.

Requirements:
  - transformer_lens
  - Plotly
  - A local or remote environment with S3 access (via S3ModelLoader)
  - The epsilon_transformers codebase
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
    model_type,
    get_activations,
    get_sweep_type
)

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

def get_layer_mses_for_activations(acts: torch.Tensor, ground_truth_beliefs: torch.Tensor) -> list[float]:
    """
    For a 4D acts tensor [n_layers, batch, seq, d_model], compute MSE for each layer
    plus the concatenation of all layers (layer_index = n_layers).
    Returns a list of length (n_layers + 1).
    """
    n_layers = acts.shape[0]
    mses = []

    # Each layer
    for i in range(n_layers):
        mse_i = run_activation_to_beliefs_regression_fast(acts[i], ground_truth_beliefs)
        mses.append(float(mse_i))

    # "All-layer" concat
    all_layers_acts = acts.permute(1, 2, 0, 3).reshape(acts.shape[1], acts.shape[2], -1)
    mse_concat = run_activation_to_beliefs_regression_fast(all_layers_acts, ground_truth_beliefs)
    mses.append(float(mse_concat))

    return mses


def random_init_control_experiment(
    transformer_config: HookedTransformerConfig,
    nn_inputs: torch.Tensor,
    nn_beliefs: torch.Tensor,
    n_random_inits: int,
) -> pd.DataFrame:
    """
    Builds n_random_inits random-initialized models for the same architecture.
    For each seed:
      - forward pass => get_activations
      - compute layer MSE + concat MSE

    Returns a DataFrame with columns:
      [
        "seed", "layer_index", "MSE",
        "random_or_trained"="random",  (we’ll add norm_type outside)
      ]
    """
    # Build a dummy model to see number of layers
    dummy_model = HookedTransformer(transformer_config)
    with torch.no_grad():
        dummy_acts = get_activations(dummy_model, nn_inputs, "transformer")
    n_layers = dummy_acts.shape[0]

    records = []
    for seed in tqdm(range(n_random_inits), desc="Random seeds"):
        # 1) Create random model
        transformer_config.seed = seed
        random_model = HookedTransformer(transformer_config)

        # 2) Forward pass
        with torch.no_grad():
            r_acts = get_activations(random_model, nn_inputs, "transformer")

        # 3) MSE per layer
        mses_this_seed = get_layer_mses_for_activations(r_acts, nn_beliefs)
        # Length: n_layers+1
        for layer_idx, mse_val in enumerate(mses_this_seed):
            records.append({
                "seed": seed,
                "layer_index": layer_idx,  # last = concat
                "MSE": mse_val,
                "random_or_trained": "random"
            })

    df = pd.DataFrame(records)
    return df


def shuffle_model_weights(model: HookedTransformer) -> HookedTransformer:
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


def plot_run_figure(
    df_run: pd.DataFrame,
    run_id: str,
    outpath_html: str
):
    """
    Create a figure for a single run that shows:
     - MSE vs. checkpoint (for trained and control models)
     - MSE vs. seed (for random)
     - For each layer_index
     - For norm/unnorm beliefs
    Saves an interactive HTML to outpath_html.

    We assume df_run has columns:
      [checkpoint, norm_type, random_or_trained, seed, layer_index, MSE]
    """
    fig = go.Figure()
    # Group by [norm_type, random_or_trained, layer_index]
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )
    pyo.plot(fig, filename=outpath_html, auto_open=False)
    print(f"Saved figure to {outpath_html}")


#%% [BLOCK 2]: Main Analysis Over All Runs (Immediate Save)
"""
We'll pick a single sweep. For each run:
  1. Load MSP data (norm + unnorm).
  2. Random-init (norm + unnorm) -> keep all seeds.
  3. For each checkpoint:
       - Compute MSE for:
             * the trained model,
             * the trained model with shuffled weights (control).
       - Do this for both (normalized) and (unnormalized) beliefs.
  4. Combine into a run-specific DataFrame.
  5. Save run-specific CSV.
  6. Save figure.
"""

loader = S3ModelLoader()

sweeps = {
    "20241205175736": "Transformer"
}
sweep_id = list(sweeps.keys())[0]

runs = loader.list_runs_in_sweep(sweep_id)
print(f"Sweep {sweep_id} has runs:", runs)

# Make a local cache folder for each run
base_outdir = f"analysis_cache/{sweep_id}"
os.makedirs(base_outdir, exist_ok=True)

for run_id in runs:
    print(f"\n=== Analyzing run: {run_id} ===")
    run_outdir = os.path.join(base_outdir, run_id)
    os.makedirs(run_outdir, exist_ok=True)

    # If we see a "run_data.csv" already, skip
    run_csv_path = os.path.join(run_outdir, "run_data.csv")
    if os.path.exists(run_csv_path):
        print(f"Already found {run_csv_path}, skipping re-analysis.")
        continue

    # 1) List checkpoints
    all_ckpts = loader.list_checkpoints(sweep_id, run_id)
    if not all_ckpts:
        print("No checkpoints found; skipping.")
        continue

    # 2) Load a sample checkpoint to get model cfg + MSP data
    sample_ckpt = all_ckpts[0]
    device = "cpu"
    model, run_config = loader.load_checkpoint(sweep_id, run_id, sample_ckpt, device=device)
    if model_type(model) != "transformer":
        print(f"Skipping run {run_id}; not a transformer.")
        continue

    # Prepare MSP data
    nn_inputs, nn_beliefs, _, _, nn_unnormalized_beliefs = prepare_msp_data(
        run_config, run_config["model_config"]
    )
    nn_inputs = nn_inputs.to(device)
    nn_beliefs = nn_beliefs.to(device)
    nn_unnormalized_beliefs = nn_unnormalized_beliefs.to(device)

    # 3) Random init for normal
    n_seeds = 10
    transformer_config = model.cfg
    print(f"Random-init control (normalized) for run={run_id} ...")
    df_random_norm = random_init_control_experiment(
        transformer_config, nn_inputs, nn_beliefs, n_seeds
    )
    df_random_norm["norm_type"] = "normalized"
    df_random_norm["checkpoint"] = "RANDOM"
    df_random_norm["sweep_id"] = sweep_id
    df_random_norm["run_id"] = run_id

    # 4) Random init for unnormalized
    print(f"Random-init control (unnormalized) for run={run_id} ...")
    df_random_unnorm = random_init_control_experiment(
        transformer_config, nn_inputs, nn_unnormalized_beliefs, n_seeds
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
            acts_ckpt = get_activations(model_ckpt, nn_inputs, "transformer")

        # MSE for the trained model:
        mses_norm = get_layer_mses_for_activations(acts_ckpt, nn_beliefs)
        mses_unnorm = get_layer_mses_for_activations(acts_ckpt, nn_unnormalized_beliefs)
        n_layers = acts_ckpt.shape[0]
        for layer_idx in range(n_layers+1):
            # For normalized beliefs
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
            # For unnormalized beliefs
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

        # --- New: Shuffled weights control ---
        # Create a shuffled copy of the model.
        model_ckpt_shuffled = shuffle_model_weights(model_ckpt)
        with torch.no_grad():
            acts_ckpt_shuffled = get_activations(model_ckpt_shuffled, nn_inputs, "transformer")
        mses_norm_shuffled = get_layer_mses_for_activations(acts_ckpt_shuffled, nn_beliefs)
        mses_unnorm_shuffled = get_layer_mses_for_activations(acts_ckpt_shuffled, nn_unnormalized_beliefs)
        for layer_idx in range(n_layers+1):
            # For normalized beliefs with shuffled weights
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
            # For unnormalized beliefs with shuffled weights
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

    # Combine into single DF
    df_run = pd.concat([df_random_norm, df_random_unnorm, df_ckpt], ignore_index=True)
    print(f"Shape of final DF for run {run_id}: {df_run.shape}")

    # 6) Save run CSV
    df_run.to_csv(run_csv_path, index=False)
    print(f"Saved run data to {run_csv_path}")

    # 7) Save figure
    #    We'll produce a single figure: MSE vs. checkpoint/seed, color-coded by layer, norm_type, random_or_trained.
    #figure_outpath = os.path.join(run_outdir, "mse_plot.html")
    #plot_run_figure(df_run, run_id, figure_outpath)

    print(f"Analysis done for run {run_id} -- you can now inspect the CSV and HTML figure.")


print("=== Done analyzing all runs. ===")
# %%
