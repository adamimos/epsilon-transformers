# analysis_single_run_blocks.py
"""
analysis_single_run_blocks.py

Block-by-block script demonstrating:
1. Selecting a sweep from a dictionary.
2. Loading a single run (by index) and a single checkpoint from S3.
3. Computing the usual MSE of the belief-state geometry in the residual stream.
4. As a control, comparing that MSE to N=100 random initializations with the same architecture.

You can run each #%% block in sequence within an IDE like VSCode or a Jupyter environment.
"""

#%% [BLOCK 1]: Imports
import sys
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformerConfig, HookedTransformer
from sklearn.linear_model import LinearRegression

# Local modules (adjust paths as needed)
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import (
    prepare_msp_data,
    run_activation_to_beliefs_regression_fast,
    model_type,
    get_activations,
    get_sweep_type
)
from tqdm.auto import tqdm
#%% [BLOCK 2]: Define sweeps and pick a run
"""
We create a dictionary of sweeps. Each key is a sweep_id (which you might interpret 
as a folder in your S3 bucket), and the value is a label or type (e.g., 'Transformer', 'RNN').

We pick one sweep from the dictionary, load its runs from S3, and index into them.
"""

# Dictionary of sweeps {sweep_id: sweep_type}
sweeps = {
    '20241205175736': 'Transformer',
    # Add more sweeps here if desired
}

# Create an S3ModelLoader instance
loader = S3ModelLoader()

# Choose which sweep we'll analyze (just pick the first key for demonstration)
sweep_id = list(sweeps.keys())[0]  # e.g. '20241205175736'
sweep_type = sweeps[sweep_id]
print(f"Selected sweep_id={sweep_id}, sweep_type={sweep_type}")

# List runs in this sweep
runs = loader.list_runs_in_sweep(sweep_id)
print("Runs found in sweep:", runs)

# Let's pick a run by index. For demonstration, we choose index 0 (the first run).
run_index = 0
run_id = runs[run_index]
print(f"Selected run: {run_id}")

#%% [BLOCK 3]: Load a single checkpoint and get MSP data
"""
- We list the available checkpoints for our chosen run.
- We pick one checkpoint (by index, for example).
- We load that checkpoint into memory.
- We also load MSP data for that run (model inputs, beliefs, etc.).
"""

# List checkpoints
checkpoints = loader.list_checkpoints(sweep_id, run_id)
print("Checkpoints found:", checkpoints)

# Pick a checkpoint index (let's choose the last one here).
checkpoint_index = -1
checkpoint_key = checkpoints[checkpoint_index]
print(f"Selected checkpoint: {checkpoint_key}")

# Load the model from S3
device = 'cpu'  # or 'cuda' if you have a GPU
model, run_config = loader.load_checkpoint(sweep_id, run_id, checkpoint_key, device=device)
print("Model loaded successfully.")

# Determine model and run types
nn_type = model_type(model)  # "transformer" or "rnn"
inferred_sweep_type = get_sweep_type(run_id)  # e.g. 'tom', 'fanizza', etc.

# Prepare MSP data
print("Loading MSP data...")
nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = prepare_msp_data(
    run_config, run_config["model_config"]
)
print("MSP data loaded.")

#%% [BLOCK 4]: Compute the "normal MSE" for the trained model
"""
We:
1. Get the activations (e.g., final residual stream).
2. Run a linear regression to predict beliefs from these activations.
3. Compute the MSE.
"""

print("Computing normal MSE from trained model...")

# 1. Gather activations
#    We'll only gather the final or near-final residual stream
try:
    with torch.no_grad():
        acts = get_activations(model, nn_inputs.to(device), nn_type)
except Exception as e:
    print(f"Error while getting activations: {e}")
    sys.exit(1)

n_layers = acts.shape[0]
for i in range(n_layers):
    mse = run_activation_to_beliefs_regression_fast(acts[i], nn_beliefs)
    print(f"MSE for layer {i}: {mse:.6f}")

# now run on concat of all layers
all_layers_acts = acts.permute(1,2,0,3).reshape(acts.shape[1], acts.shape[2], -1)
mse = run_activation_to_beliefs_regression_fast(all_layers_acts, nn_beliefs)
print(f"MSE for concat of all layers: {mse:.6f}")


#%% [BLOCK 5]: Control Experiment – N=100 random initializations
"""
We create 100 new models with the same architecture but random weights, 
compute the same regression MSE, and compare.
"""

if nn_type == "transformer":
    # Retrieve the config
    transformer_config = model.cfg
else:
    print("Random init control for RNN not implemented here.")
    random_mses = []
    # We will skip the random init part if it's not a transformer
    pass

if nn_type == "transformer":
    n_random_inits = 100
    random_mses = []

    print("Running control experiment with 100 random inits...")
    device = 'cpu'
    nn_inputs = nn_inputs.to(device)
    bar = tqdm(range(n_random_inits), desc=f"Random inits (mean MSE: {np.mean(random_mses) if random_mses else 0:.6f})")
    for i in bar:
        # 1. Create new random model
        transformer_config.seed = i
        transformer_config.device = device
        random_model = HookedTransformer(transformer_config)

        # 2. Forward pass to get final_acts
        with torch.no_grad():
            r_acts = get_activations(random_model, nn_inputs, "transformer")
            r_final_acts = r_acts[-1].to("cpu")

        # 3. Regress
        rand_mse = run_activation_to_beliefs_regression_fast(r_final_acts, nn_beliefs)

        random_mses.append(rand_mse.item())
        bar.set_description(f"Random inits MSE: {rand_mse:.6f}")
#%% [BLOCK 6]: Compare and plot
"""
Plot a histogram of random-init MSEs and mark the trained model's MSE for comparison.
"""

if nn_type == "transformer" and len(random_mses) > 0:
    random_mses = np.array(random_mses)
    mean_rand_mse = random_mses.mean()
    std_rand_mse  = random_mses.std()

    print("\n=== Random Inits Statistics ===")
    print(f"Mean MSE: {mean_rand_mse:.6f}")
    print(f"Std  MSE : {std_rand_mse:.6f}")

    # Plot
    plt.figure(figsize=(8,5))
    plt.hist(random_mses, bins=15, alpha=0.7, color="gray", label="Random Inits MSE")
    plt.axvline(x=mse, color="red", linestyle="--", label=f"Trained Model MSE={mse:.4f}")
    plt.title(f"Belief Regression MSE\nTrained vs. Random Inits (N=100)")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
else:
    print("Skipped plotting random init distribution (not a Transformer or no random_mses).")

print("Done!")

# %%
