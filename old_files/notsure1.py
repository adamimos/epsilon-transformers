# analysis_single_run_blocks.py
"""
analysis_single_run_blocks.py

This script does the following for a single run:
1) Creates a control set by randomly initializing N transformers of the same architecture
   and computing MSE for each layer (and the concatenated layers). This is done ONCE per run.
2) For every checkpoint in the run, computes the same MSE for each layer and the concatenation.
3) Plots a graph with:
   - X-axis = checkpoint index
   - Y-axis = MSE
   - A line (or points) for each layer's MSE across checkpoints
   - A semi-transparent band for each layer showing random-init mean ± std.

We assume we have these local modules:
  - S3ModelLoader in load_data.py
  - prepare_msp_data, get_activations, model_type, get_sweep_type, and a FAST regression method in activation_analysis.py
"""

#%% [BLOCK 1]: Imports & Helper Functions
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformer_lens import HookedTransformerConfig, HookedTransformer
from sklearn.linear_model import LinearRegression

# Local modules (adjust as needed)
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import (
    prepare_msp_data,
    run_activation_to_beliefs_regression_fast,  # The "fast" version
    model_type,
    get_activations,
    get_sweep_type
)

def get_layer_mses_for_activations(acts: torch.Tensor, ground_truth_beliefs: torch.Tensor):
    """
    For a 4D acts tensor [n_layers, batch, seq, d_model], compute:
        - MSE for each layer i in [0..n_layers-1]
        - MSE for the concatenation of all layers
    Returns a list of MSEs for each layer plus one more for 'all-layers'.
    """
    n_layers = acts.shape[0]
    mses = []

    # For each layer
    for i in range(n_layers):
        mse_i = run_activation_to_beliefs_regression_fast(acts[i], ground_truth_beliefs)
        mses.append(mse_i)

    # For "all-layers" concat
    # shape: [n_layers, batch, seq, d_model] => [batch, seq, n_layers*d_model]
    all_layers_acts = acts.permute(1,2,0,3).reshape(acts.shape[1], acts.shape[2], -1)
    mse_concat = run_activation_to_beliefs_regression_fast(all_layers_acts, ground_truth_beliefs)
    mses.append(mse_concat)

    return mses  # length = n_layers + 1

def random_init_control_experiment(
    transformer_config,
    nn_inputs: torch.Tensor,
    nn_beliefs: torch.Tensor,
    n_random_inits: int = 100
):
    """
    Runs the random-initialization control experiment:
      - For each of n_random_inits, create a fresh HookedTransformer w/ that config
      - Collect final activations for all layers (via get_activations)
      - Compute MSE for each layer + concat
    Returns:
      control_mean_mses: list of length (n_layers+1) for the mean across random inits
      control_std_mses:  list of length (n_layers+1) for the std across random inits
    """
    # First, build one "dummy" model to see how many layers there are
    dummy_model = HookedTransformer(transformer_config)
    # Grab dummy activations to get shape
    with torch.no_grad():
        dummy_acts = get_activations(dummy_model, nn_inputs, "transformer")
    n_layers = dummy_acts.shape[0]  # (n_layers, batch, seq, hidden)

    # We'll store MSE arrays of shape [n_random_inits, n_layers+1]
    all_mses = []

    for seed in tqdm(range(n_random_inits), desc="Random inits"):
        # 1. Create a random model
        transformer_config.seed = seed
        random_model = HookedTransformer(transformer_config)

        # 2. Forward pass => activations
        with torch.no_grad():
            r_acts = get_activations(random_model, nn_inputs, "transformer")
        # 3. Compute MSE for each layer + concat
        mses_for_seed = get_layer_mses_for_activations(r_acts, nn_beliefs)
        all_mses.append(mses_for_seed)

    all_mses = np.array(all_mses)  # shape [n_random_inits, n_layers+1]

    # Means and stds
    control_mean_mses = all_mses.mean(axis=0)
    control_std_mses  = all_mses.std(axis=0)

    return control_mean_mses, control_std_mses

def plot_checkpoint_mses_with_control(
    checkpoints,
    ckpt_layer_mses,      # shape [n_ckpts, n_layers+1]
    control_mean_mses,    # shape [n_layers+1]
    control_std_mses,     # shape [n_layers+1]
    layer_names=None
):
    """
    Plot a single figure with:
      - x-axis = checkpoint index (or number)
      - y-axis = MSE
      - lines for each layer's MSE across checkpoints
      - a band or errorbars for control mean ± std

    Args:
      checkpoints        (list of str): e.g. ['ckpt_0.pt', 'ckpt_1.pt', ...]
      ckpt_layer_mses    (np.ndarray): shape [n_ckpts, n_layers+1], each row is MSE for each layer + concat
      control_mean_mses  (np.ndarray): shape [n_layers+1]
      control_std_mses   (np.ndarray): shape [n_layers+1]
      layer_names (list[str]): optional, length n_layers+1.
    """
    n_ckpts, n_layers_plus_1 = ckpt_layer_mses.shape
    x = np.arange(n_ckpts)

    # If no layer names provided, generate them
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(n_layers_plus_1-1)] + ["All-layers concat"]

    plt.figure(figsize=(10, 6))

    # Plot each layer's MSE across checkpoints
    for layer_idx in range(n_layers_plus_1):
        # Extract MSE across all ckpts
        layer_mses_ckpts = ckpt_layer_mses[:, layer_idx]
        # Plot
        plt.plot(x, layer_mses_ckpts, marker='o', label=layer_names[layer_idx])

        # Overplot the control mean ± std as a horizontal band
        mean_val = control_mean_mses[layer_idx]
        std_val  = control_std_mses[layer_idx]
        # We'll just do a horizontal band. Another approach is to do errorbars at each x.
        plt.fill_between(
            x,
            mean_val - std_val,
            mean_val + std_val,
            alpha=0.2
        )
        # Or you might want to do a horizontal line for the mean:
        plt.axhline(mean_val, linestyle='--', alpha=0.2)

    plt.xlabel("Checkpoint Index")
    plt.ylabel("MSE")
    plt.title("Layer MSE vs. Checkpoint (with Random Init Control ± STD)")
    plt.legend()
    # Replace x-ticks with checkpoint filenames (truncated if you want)
    plt.xticks(x, [ckpt.split('/')[-1] for ckpt in checkpoints], rotation=45)
    plt.tight_layout()
    plt.show()

#%% [BLOCK 2]: Define sweeps and pick a run
"""
We create a dictionary of sweeps. Each key is a sweep_id (S3 folder).
We pick one sweep from the dictionary, load its runs from S3, and index into them.
"""

sweeps = {
    '20241205175736': 'Transformer'
}

loader = S3ModelLoader()

# Choose sweep
sweep_id = list(sweeps.keys())[0]
sweep_type = sweeps[sweep_id]
print(f"Selected sweep_id={sweep_id}, sweep_type={sweep_type}")

# List runs
runs = loader.list_runs_in_sweep(sweep_id)
print("Runs found:", runs)

run_index = 0
run_id = runs[run_index]
print(f"Selected run: {run_id}")

#%% [BLOCK 3]: Load MSP data once (for the entire run) + control experiment
"""
1) We'll load the run config from the run (we can pick the first checkpoint or run_config).
2) Prepare MSP data (nn_inputs, nn_beliefs, etc.).
3) We'll do the random init control once, returning mean/std MSE for each layer.
"""

# For convenience, just pick the first checkpoint to load the run_config
all_checkpoints = loader.list_checkpoints(sweep_id, run_id)
if not all_checkpoints:
    raise ValueError("No checkpoints found for this run!")

# We'll pick the first checkpoint to get the model architecture
sample_ckpt = all_checkpoints[0]
device = "cpu"
model, run_config = loader.load_checkpoint(sweep_id, run_id, sample_ckpt, device=device)
nn_type = model_type(model)
if nn_type != "transformer":
    raise ValueError(f"This script only covers Transformer-based runs. Found nn_type={nn_type}.")

# Load MSP data
print("Loading MSP data...")
nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = prepare_msp_data(
    run_config, run_config["model_config"]
)
print("MSP data loaded.  Shape:", nn_inputs.shape, nn_beliefs.shape)

transformer_config = model.cfg  # We'll reuse this for random init

# Move data to CPU (or GPU if you'd like)
nn_inputs = nn_inputs.to(device)
nn_beliefs = nn_beliefs.to(device)

# Random init control
n_random_inits = 50  # For demonstration, 50 random seeds
print(f"Running random-init control with N={n_random_inits} for this run...")
control_mean_mses, control_std_mses = random_init_control_experiment(
    transformer_config, nn_inputs, nn_beliefs, n_random_inits=n_random_inits
)
print("Finished random-init control.")

#%% [BLOCK 4]: Analyze MSE for every checkpoint
"""
For each checkpoint in the run:
  1) Load the checkpoint
  2) Get activations
  3) Compute MSE for each layer + concat
Collect these in a list or array that we can plot.
"""

layer_mse_results = []  # will be shape [n_ckpts, n_layers+1]
ckpt_names = []

for ckpt in tqdm(all_checkpoints, desc="Analyzing checkpoints"):
    # Load model
    model, run_config_ckpt = loader.load_checkpoint(sweep_id, run_id, ckpt, device=device)
    # Get activations
    with torch.no_grad():
        acts = get_activations(model, nn_inputs, "transformer")
    # Compute MSE per layer
    mses_layer = get_layer_mses_for_activations(acts, nn_beliefs)  # returns list [layer0, layer1, ..., concat]
    layer_mse_results.append(mses_layer)
    ckpt_names.append(ckpt)

layer_mse_results = np.array(layer_mse_results)  # shape [n_ckpts, n_layers+1]

#%% [BLOCK 5]: Plot the results
"""
We have:
  - ckpt_names: list of checkpoint keys
  - layer_mse_results: shape [n_ckpts, n_layers+1]
  - control_mean_mses, control_std_mses: shape [n_layers+1]
We want a single figure with lines for each layer's MSE across ckpts, and
a semi-transparent band or errorbars for the random init control.

You can choose whether to show all layers in one plot or multiple subplots. 
Below, we put them all in one plot for simplicity.
"""

# For naming layers, let's discover number of layers from shape
n_layers_plus_1 = layer_mse_results.shape[1]
# We'll name them "Layer 0", "Layer 1", ... and "All-layers" for the last index
layer_names = [f"Layer {i}" for i in range(n_layers_plus_1-1)] + ["All-layers concat"]

plot_checkpoint_mses_with_control(
    ckpt_names,
    layer_mse_results,
    control_mean_mses,
    control_std_mses,
    layer_names=layer_names
)

print("Done!")
# %%

