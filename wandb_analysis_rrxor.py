# type: ignore
# %%
import torch
import wandb
import numpy as np
from epsilon_transformers import build_network
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# %%
import wandb

from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
    collect_path_probs_with_paths,
    collect_paths_with_beliefs,
)

# %%
from typing import List, Dict, Optional, Tuple
import yaml
from pydantic import BaseModel
from epsilon_transformers.comp_mech.processes import (
    random_random_xor,
    zero_one_random,
    mess3,
)
from epsilon_transformers.configs import SweepConfig
import torch
import torch.nn.functional as F
import wandb
import torch.nn as nn

from epsilon_transformers import (
    build_dataset,
    build_optimizer,
    build_network,
    create_validation_set,
    build_probabilistic_dataset,
)
from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
)

# %%


def load_model_artifact(
    user_or_org,
    project_name,
    artifact_name,
    artifact_type,
    artifact_version,
    config,
    device="cpu",
):
    api = wandb.Api()
    artifact_reference = (
        f"{user_or_org}/{project_name}/{artifact_name}:{artifact_version}"
    )
    print(f"Loading artifact {artifact_reference}")
    artifact = wandb.use_artifact(artifact_reference, type=artifact_type)
    artifact_dir = artifact.download()
    artifact_file = (
        f"{artifact_dir}/{artifact_name}.pt"  # Making the filename programmatic
    )
    model = build_network(config, torch.device(device))
    model.load_state_dict(torch.load(artifact_file, map_location=device))
    return model


def prepare_data(config, device="cpu"):
    # Assume MSP_tree and process are defined globally or passed as parameters
    X_val, Y_val, val_weights = create_validation_set(MSP_tree, config["n_ctx"])
    # Convert to tensors and move to the specified device
    X_val = torch.tensor(X_val, dtype=torch.int).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
    return X_val, Y_val, val_weights


def evaluate_model(model, X_val, Y_val, val_weights, batch_size=100):
    print(f"Evaluating model on {X_val.size(0)} validation samples")
    model.eval()
    all_losses = []
    with torch.no_grad():
        for i in tqdm(range(0, X_val.size(0), batch_size)):
            X_val_batch = X_val[i : i + batch_size]
            Y_val_batch = Y_val[i : i + batch_size]
            val_weights_batch = val_weights[i : i + batch_size]
            logits = model(X_val_batch)
            logits_flat = logits.view(-1, logits.shape[-1])
            Y_val_flat = Y_val_batch.view(-1)
            loss = F.cross_entropy(logits_flat, Y_val_flat, reduction="none")
            loss = loss.view(X_val_batch.shape[0], X_val_batch.shape[1])
            weighted_loss = loss * val_weights_batch.unsqueeze(1).repeat(
                1, X_val_batch.shape[1]
            )
            all_losses.append(weighted_loss)
    return torch.cat(all_losses, dim=0)


def run_linear_regression(activations, beliefs, weights=None):
    reg = LinearRegression().fit(activations, beliefs, sample_weight=weights)
    predicted_beliefs = reg.predict(activations)
    return reg, predicted_beliefs


def fetch_run_config(user_or_org, project_name, run_id):
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/{run_id}"
    run = api.run(run_path)
    return run.config


def extract_activations_with_cache(model, X_val, device, batch_size=100):
    """
    Extract activations from the model for the validation set, utilizing a caching mechanism.

    Args:
    - model: The trained model from which to extract activations.
    - X_val: The validation dataset.
    - device: The device on which the model is running.
    - batch_size: The size of batches to process the validation dataset.

    Returns:
    - activations: The extracted activations from the model.
    """
    print(f"Extracting activations for {X_val.size(0)} validation samples")

    model.eval()
    activations = []
    with torch.no_grad():
        for i in tqdm(range(0, X_val.size(0), batch_size)):
            X_val_batch = X_val[i : i + batch_size].to(device)
            # Utilize the model's caching mechanism during forward pass
            _, acts = model.run_with_cache(X_val_batch)
            # Assuming 'ln_final.hook_normalized' is the key for the activations we're interested in
            #acts = acts["ln_final.hook_normalized"]  # [batch_size, n_ctx, d_model]
            acts0 = acts['blocks.0.ln1.hook_normalized'] # [batch_size, n_ctx, d_model]
            acts1 = acts['blocks.1.ln1.hook_normalized']
            acts2 = acts['blocks.2.ln1.hook_normalized']
            acts3 = acts['blocks.3.ln1.hook_normalized']

            # stack acts on last dim so that we get [batch_size, n_ctx, d_model*4]
            acts = torch.cat([acts0, acts1, acts2, acts3], dim=-1)
            #acts = acts2
            print(acts.shape)
            activations.append(acts.cpu())

    activations = torch.cat(activations, dim=0)  # Concatenate all batch activations
    return (
        activations.numpy()
    )  # Convert to numpy array for consistency with the rest of the codebase


def precompute_activation_and_belief_inds(MSP_tree, n_ctx, X_val):
    """
    X_val: [n_samples, n_ctx]
    n_ctx: int
    """
    all_beliefs = np.zeros((X_val.size(0), n_ctx, 5))

    for i in tqdm(range(n_ctx)):
        results = collect_paths_with_beliefs(MSP_tree, i + 1)
        # Convert sequences and beliefs into a dictionary for faster lookup
        seqs_to_beliefs = {tuple(s[0]): s[2] for s in results}
        # Pre-convert all X_val sequences up to i+1 into tuples for faster batch processing
        x_tuples = [tuple(x[: i + 1].tolist()) for x in X_val]
        for j, x_tuple in enumerate(x_tuples):
            if x_tuple in seqs_to_beliefs:
                all_beliefs[j, i, :] = seqs_to_beliefs[x_tuple]

    return all_beliefs


def collect_activations_and_beliefs(activations, X_val, MSP_tree, config):
    # activations: [n_samples, n_ctx, d_model]
    # X_val: [n_samples, n_ctx]
    print(f"Collecting activations and beliefs for {X_val.size(0)} validation samples")
    all_acts = []
    all_beliefs = []
    for i in range(config["n_ctx"]):
        acts = activations[:, i, :]  # [n_samples, d_model]
        results = collect_paths_with_beliefs(MSP_tree, i + 1)
        seqs = [s[0] for s in results]
        beliefs = [s[2] for s in results]
        seqs_beliefs = []
        for x in tqdm(X_val[:, : i + 1]):
            # convert to tuple
            x = tuple(x.tolist()) if type(x.tolist()) == list else (x.tolist(),)
            # find the index of x in seqs
            idx = seqs.index(x)
            seqs_beliefs.append(beliefs[idx])
        seqs_beliefs = np.array(seqs_beliefs)
        all_acts.append(acts)
        all_beliefs.append(seqs_beliefs)

    all_acts = np.concatenate(all_acts, axis=0)  # [n_samples * n_ctx, d_model]
    all_beliefs = np.concatenate(all_beliefs, axis=0)  # [n_samples * n_ctx, 3]
    return all_acts, all_beliefs


def project_to_simplex(points):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    # Assuming points is a 2D array with shape (n_points, 3)
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y


# %%

import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import pandas as pd
from PIL import Image

def create_dataframe(points, beliefs):
    x, y = project_to_simplex(np.array(points))
    return pd.DataFrame({
        "x": x,
        "y": y,
        "r": beliefs[:, 0],
        "g": beliefs[:, 1],
        "b": beliefs[:, 2],
    })
def percentile_agg(array, q=50):
    """Custom aggregation function to compute the percentile."""
    return np.percentile(array, q)


def aggregate_data(df):
    cvs = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(-0.1, 1.1), y_range=(-0.1, np.sqrt(3) / 2 + 0.1))
    agg_funcs = {"r": ds.mean("r"), "g": ds.mean("g"), "b": ds.mean("b")}
    return {color: cvs.points(df, "x", "y", agg_funcs[color]) for color in ["r", "g", "b"]}

def custom_thresholded_linear_cmap(threshold, color):
    black_rgb = (0, 0, 0)
    rgb_dict = {'black': black_rgb, 'green': (0, 255, 0), 'red': (255, 0, 0), 'blue': (0, 0, 255)}
    cmap = []
    # we want to interpolate from black to color,
    # over the range above the threshold
    for v in np.linspace(0, 1, 1000):
        if v < threshold:
            cmap.append(black_rgb)
        else:
            interp_color = tuple(np.array(black_rgb) + (np.array(rgb_dict[color]) - np.array(black_rgb)) * ((v - threshold) / (1 - threshold)))
            cmap.append(interp_color)
    return cmap

def combine_channels_to_rgb(agg_r, agg_g, agg_b):
    #green_cmap = custom_thresholded_linear_cmap(0.75,'green')
    
    img_r = tf.shade(agg_r, cmap=["black", "red"], how="linear")
    img_g = tf.shade(agg_g, cmap=["black", "green"], how="linear")
    img_b = tf.shade(agg_b, cmap=["black", "blue"], how="linear")

    img_r = tf.spread(img_r, px=1, shape="circle")
    img_g = tf.spread(img_g, px=1, shape="circle")
    img_b = tf.spread(img_b, px=1, shape="circle")

    r_array = np.array(img_r.to_pil()).astype(np.float64)
    g_array = np.array(img_g.to_pil()).astype(np.float64)
    b_array = np.array(img_b.to_pil()).astype(np.float64)

    rgb_image = np.stack([r_array[:, :, 0], g_array[:, :, 1], b_array[:, :, 2]], axis=-1)
    return Image.fromarray(np.uint8(rgb_image))

def generate_image(beliefs, all_beliefs):
    df = create_dataframe(beliefs, all_beliefs)
    agg_data = aggregate_data(df)
    return combine_channels_to_rgb(agg_data["r"], agg_data["g"], agg_data["b"])

def generate_belief_state_figures_datashader(belief_states, all_beliefs, predicted_beliefs, gt_fig=None, plot_triangles=False):
    if gt_fig is None:
        img_gt = generate_image(belief_states, belief_states)  # Using the same data for ground truth visualization
    else:
        img_gt = gt_fig
    img_pb = generate_image(predicted_beliefs, all_beliefs)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, facecolor="black")
    for ax in axs:
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    axs[0].imshow(img_gt)
    axs[1].imshow(img_pb)

    axs[0].axis("off")
    axs[1].axis("off")

    title_y_position = -0.1
    fig.text(0.5, title_y_position, "Ground Truth", ha="center", va="top", transform=axs[0].transAxes, color="white", fontsize=15)
    fig.text(0.5, title_y_position, "Residual Stream", ha="center", va="top", transform=axs[1].transAxes, color="white", fontsize=15)

    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3) / 2, 0, 0], "white", lw=2)

    return fig


def generate_belief_state_figures(
    belief_states,
    all_beliefs,
    predicted_beliefs,
    plot_triangles=False,
    alpha=0.5,
    s=0.01,
    show_plot=False,
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Ground Truth
    colors_gt = belief_states
    x_gt, y_gt = project_to_simplex(np.array(belief_states))

    # randomly permute the rows
    axs[0].scatter(x_gt, y_gt, c=colors_gt, s=s, alpha=1.0, marker=',')
    axs[0].set_title("Ground Truth", pad=-20)
    axs[0].axis("off")

    # Residual Stream
    colors_rs = all_beliefs  # [n_samples * n_ctx, 3]
    x_rs, y_rs = project_to_simplex(np.array(predicted_beliefs))
    axs[1].scatter(x_rs, y_rs, c=colors_rs, s=s, alpha=alpha)
    axs[1].set_title("Residual Stream", pad=-20)
    axs[1].axis("off")

    # Plot the equilateral triangle if requested
    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3) / 2, 0, 0], "k-")

    # set x y limits
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_ylim(-0.1, 1.1)
    # Close the plot if not intended to show
    if not show_plot:
        plt.close(fig)
    # Return the figure object for further manipulation or saving
    return fig


# %%


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")
wandb.init()
user_or_org = "adamimos"
project_name = "transformer-MSPs"
# run_id = '2zulyhrv' # mess3 param change
# run_id = 's6p0aaci' # zero one random
#run_id = "halvkdvk"  # mess3 param change long run
#run_id = 'anmsm2sv' # mess 3 original param, with scheduler
#run_id = "0q4iok4y" #rrxor
#run_id = "rnyh5kb9" #rrxor
run_id = "0k7hf4nl" # rrxor
run_id = "vfs4q106" # rrxor
# make a folder in figures with the name of the run_id
# for the images
if not os.path.exists(f"figures/{run_id}"):
    os.makedirs(f"figures/{run_id}")

# %%


def fetch_artifacts_for_run(user_or_org, project_name, run_id):
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
    run = api.run(run_path)
    artifacts = run.logged_artifacts()
    return artifacts


arts = fetch_artifacts_for_run(user_or_org, project_name, run_id)
print(f"the number of artifacts is {len(arts)}")
# %%
config = fetch_run_config(user_or_org, project_name, run_id)
#process = mess3(0.05, 0.85)
# process = zero_one_random()
# process = mess3()
if run_id == "0q4iok4y" or run_id == "rnyh5kb9" or run_id == "0k7hf4nl" or run_id == "vfs4q106":
    process = random_random_xor()
else:
    process = mess3()
MSP_tree = mixed_state_tree(process, config["n_ctx"] + 1)
belief_states = MSP_tree.get_belief_states()
belief_states = np.array(belief_states)
X_val, Y_val, val_weights = create_validation_set(MSP_tree, config["n_ctx"])

X_val = torch.tensor(X_val, dtype=torch.int).to(device)
val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
val_weights_repeated = val_weights.unsqueeze(1).repeat(1, config["n_ctx"]).cpu().numpy()
Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
print(f"X_val: {X_val.size()}, Y_val: {Y_val.size()}, val_weights: {val_weights.size()}, val_weights_repeated: {val_weights_repeated.shape}")
img_gt = generate_image(belief_states, belief_states)

# Assign a group label to each belief state row based on uniqueness with a tolerance for L2 distance
def assign_labels_to_belief_states(arr, tol=0.01):
    unique = []
    labels = []
    for row in arr:
        distances = np.linalg.norm(np.array(unique) - row, axis=1) if unique else [float('inf')]
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] > tol:  # if the closest distance is greater than tol, it's unique
            unique.append(row)
            labels.append(len(unique) - 1)  # Assign a new label for this unique row
        else:
            labels.append(min_dist_idx)  # Assign the label of the closest existing unique row
    return np.array(labels)

belief_state_labels = assign_labels_to_belief_states(belief_states)
print(f"Number of unique rows with tolerance: {len(np.unique(belief_state_labels))}")
print(f"Belief state labels: {belief_state_labels}")
unique_belief_states = belief_states[belief_state_labels]
print(f"Number of unique rows with tolerance: {len(unique_belief_states)}")
unique_labels = np.unique(belief_state_labels)
unique_belief_states = np.zeros((len(unique_labels), 5))
for ul in unique_labels:
    idx = np.where(belief_state_labels == ul)[0][0]
    unique_belief_states[ul] = belief_states[idx]

# %%
from sklearn.decomposition import PCA

# run pca
pca = PCA(n_components=3)
pca.fit(unique_belief_states)
unique_belief_states_pca = pca.transform(unique_belief_states)

# plot in 3d using plotly with 36 unique colors
import plotly.express as px
import plotly.graph_objects as go

# Generate a list of 36 unique colors
colors = px.colors.qualitative.Pastel + px.colors.qualitative.Set3 + px.colors.qualitative.Dark24

fig = go.Figure()
for idx, label in enumerate(np.unique(unique_labels)):
    filtered_data = unique_belief_states_pca[unique_labels == label]
    fig.add_trace(go.Scatter3d(x=filtered_data[:, 0], y=filtered_data[:, 1], z=filtered_data[:, 2],
                               mode='markers', name=f'Label {label}',
                               marker=dict(size=5, opacity=1.0, color=colors[idx % 36])))
fig.show()



#%%
all_beliefs = precompute_activation_and_belief_inds(MSP_tree, config["n_ctx"], X_val)
# all_beliefs is shape [n_samples, n_ctx, 5]
# X_val is shape [n_samples, n_ctx]

def assign_labels_to_belief_states(all_beliefs, belief_states, belief_state_lables):
    # for each row of all_beliefs find the row in belief_states that is closest
    # then assign the label from belief_state_labels
    labels = []
    all_belief_labels = np.zeros(all_beliefs.shape[0:2])
    for i in range(all_beliefs.shape[0]):
        for j in range(all_beliefs.shape[1]):
            all_belief_labels[i, j] = belief_state_lables[np.argmin(np.linalg.norm(belief_states - all_beliefs[i, j], axis=1))]
    return all_belief_labels

all_beliefs_labels = assign_labels_to_belief_states(all_beliefs, belief_states, belief_state_labels)
# all_belief_labels is shape [n_samples, n_ctx]

# %%
results = []
#for art in tqdm(arts):
for art in arts:
    art = arts[len(arts) - 2]
    art_name = art.name
    print(f"Artifact: {art_name}")
    
    # art.name is "name:version" so we split it to get the name and version
    artifact_name, artifact_version = art_name.split(":")
    epoch_num = int(artifact_name.split("_")[-1])

    # if epoch is not divisible by 5 then continue
    #if epoch_num % 5 != 0:
    #    continue
    if os.path.exists(f"figures/{run_id}/belief_states_{artifact_name}.png"):
        # Skip this run if the .png already exists
        continue
    def analyze_artifact(
        user_or_org,
        project_name,
        artifact_name,
        artifact_type,
        artifact_version,
        config,
        device,
        all_beliefs,
        val_weights,
        gt_fig=None,
    ):
        model = load_model_artifact(
            user_or_org,
            project_name,
            artifact_name,
            artifact_type,
            artifact_version,
            config,
            device=device,
        )

        # myopic_entropy_rate = myopic_entropy(MSP_tree)
        # minimum_cross_entropy = myopic_entropy_rate[1:]
        path_probs = collect_path_probs_with_paths(MSP_tree, config["n_ctx"])
        belief_states = MSP_tree.get_belief_states()
        belief_states = np.array(belief_states)

        activations = extract_activations_with_cache(
            model, X_val, device, batch_size=10000
        )
        # activations is shape [n_samples, n_ctx, d_model]
        # reshape all_beliefs and activations
        all_beliefs_reshaped = all_beliefs.reshape(-1, all_beliefs.shape[-1])
        all_beliefs_labels_reshaped = all_beliefs_labels.reshape(-1)
        all_acts = activations.reshape(-1, activations.shape[-1])
        val_weights_reshaped = val_weights.reshape(-1)
        reg, predicted_beliefs = run_linear_regression(all_acts, all_beliefs_reshaped, None)
        # predicted_beliefs is shape [n_samples * n_ctx, 5]
        # add the image generation here
        #reg_c = reg.coef_.T # [d_model, 3]
        #reg_b = reg.interce|pt_ # [3]
        #U_c = model.unembed.W_U.detach().cpu().numpy() # [d_model, 3]
        #U_b = model.unembed.b_U.detach().cpu().numpy() # [3]

        # determine if the reg and the U are the same or not

        #fig3 = generate_belief_state_figures(belief_states, all_beliefs_reshaped, predicted_beliefs, plot_triangles=False, alpha=0.1)
        #fig3.savefig(f"figures/{run_id}/belief_states_{artifact_name}_no_datashader.png", dpi=300)
        
        # we have a [n_samples*n_ctx, 5] array of predicted beliefs
        # we want to plot these according to the pca in 3d first lets get
        # the pcs
        predicted_beliefs_pcs = pca.transform(predicted_beliefs)
        #pca_activations = PCA(n_components=3)
        #pca_activations.fit(predicted_beliefs)
        # predicted_beliefs_pcs is shape [n_samples * n_ctx, 3]
        # now we want to plot these 3d points and color them according to the labels
        #predicted_beliefs_pcs = pca_activations.transform(predicted_beliefs)

        from plotly.subplots import make_subplots

        # Create a subplot with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                            subplot_titles=("Ground Truth", "Predicted Beliefs"),
                            shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.01)

        # Plotting the ground truth on the left (first column) without legend
        unique_labels = np.unique(all_beliefs_labels_reshaped)
        all_beliefs_pcs = pca.transform(all_beliefs_reshaped)  # Transforming the data through PCA
        for idx, label in enumerate(unique_labels):
            filtered_data = all_beliefs_pcs[all_beliefs_labels_reshaped == label]
            fig.add_trace(go.Scatter3d(x=filtered_data[:, 0], y=filtered_data[:, 1], z=filtered_data[:, 2],
                                       mode='markers', showlegend=False,
                                       marker=dict(size=5, opacity=1.0, color=colors[idx % 36])), row=1, col=1)

        # Plotting the predicted beliefs on the right (second column) without legend and with full opacity
        for idx, label in enumerate(unique_labels):
            filtered_data = predicted_beliefs_pcs[all_beliefs_labels_reshaped == label]
            fig.add_trace(go.Scatter3d(x=filtered_data[:, 0], y=filtered_data[:, 1], z=filtered_data[:, 2],
                                       mode='markers', showlegend=True,
                                       marker=dict(size=2, opacity=0.1, color=colors[idx % 36])), row=1, col=2)
            
            # now on top of that plot the center of mass with 1 opacity, bigger marker, and same color
            predicted_beliefs_filtered = predicted_beliefs[all_beliefs_labels_reshaped == label]
            com = np.mean(predicted_beliefs_filtered, axis=0)
            com_pca = pca.transform(com.reshape(1, -1))[0]
            fig.add_trace(go.Scatter3d(x=[com_pca[0]], y=[com_pca[1]], z=[com_pca[2]],
                                       mode='markers', showlegend=False,
                                       marker=dict(size=5, opacity=1.0, color=colors[idx % 36])), row=1, col=2)

        # Adjusting layout for better visualization and setting the same axis limits on both plots
# Adjusting layout for better visualization and setting the same axis limits on both plots
        axis_limits = dict(showgrid=True, zeroline=False, showticklabels=True, range=[-1, 1])
        fig.update_layout(height=600, width=1200, title_text="Ground Truth vs Predicted Beliefs Visualization",
                        scene=dict(xaxis=axis_limits, yaxis=axis_limits, zaxis=axis_limits, uirevision="some_fixed_value"),
                        scene2=dict(xaxis=axis_limits, yaxis=axis_limits, zaxis=axis_limits, uirevision="some_fixed_value"),
                        showlegend=True)
        #fig.update_scenes(xaxis_autorange=True, yaxis_autorange=True, zaxis_autorange=True)
        fig.write_html(f"figures/{run_id}/interactive_ground_truth_vs_predicted_{artifact_name}.html")


# Note: Replace "some_fixed_value" with your chosen value for uirevision, such as run_id or artifact_name, to keep the UI state synchronized.

        
        
        #fig = generate_belief_state_figures_datashader(
        #    belief_states, all_beliefs_reshaped, predicted_beliefs, plot_triangles=True, gt_fig=gt_fig
        #)
        # belief_states # [n_ctx, 3], all_beliefs_reshaped # [n_samples * n_ctx, 3], predicted_beliefs # [n_samples * n_ctx, 3]
        #fig.savefig(f"figures/{run_id}/belief_states_{artifact_name}.png", dpi=300)
        return (
            artifact_name,
            reg,
            belief_states,
            all_beliefs_reshaped,
            predicted_beliefs,
        )

    # project_name = 'zero_one_random_initial_sweep'
    result = analyze_artifact(
        user_or_org,
        'transformer-MSPs',
        artifact_name,
        art.type,
        art.version,
        config,
        device,
        all_beliefs,
        val_weights_repeated,
        img_gt,
    )
    # move results to cpu
    # results.append(result)
    # plot_belief_states(result[2], result[3], result[4])

# %%
# save the results to file
import pickle

with open("results_z1r.pkl", "wb") as f:
    pickle.dump(results, f)


# %%

# %%
# now we want to make an animation of the belief states
# we will use the results from the previous cell
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # Define a single figure and axes to animate
figs = []

for r in tqdm(results):
    figs.append(generate_belief_state_figures(r[2], r[3], r[4]))

# Prepare the frames for the animation
frames = [[plt.imshow(fig[i], animated=True)] for i in range(len(figs)) for fig in figs]

# Create an animation of the belief states using the frames generated above
ani = ArtistAnimation(fig, frames, interval=200, blit=True)

# Save the animation
ani.save("belief_states_animation.mp4", writer="ffmpeg", fps=30)


# %%
# load results
import pickle

with open("results.pkl", "rb") as f:
    results = pickle.load(f)
# %%
for i, r in tqdm(enumerate(results)):
    if i >= -1:
        fig = generate_belief_state_figures(
            r[2], r[3], r[4], plot_triangles=True, alpha=1.0, s=15
        )
        # save the figure
        fig.savefig(f"belief_states_zor_{i}.png")


# %%
import cv2
import os

# Define the codec and create VideoWriter object for AVI format
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("belief_states_movie.avi", fourcc, 10.0, (640, 480))

belief_state_images = [
    img
    for img in os.listdir()
    if img.startswith("belief_states_z") and img.endswith(".png")
]
belief_state_images.sort(
    key=lambda x: int(x.split("_")[3].split(".")[0])
)  # Sort files by the number in their name
frames = []
for img_name in tqdm(belief_state_images):
    frame = cv2.imread(img_name)
    print(frame.min())
    # does frame have any nans?
    if np.isnan(frame).any():
        print(f"Frame {img_name} has nans")
    out.write(frame)  # Write out frame to video
    frames.append(frame)

out.release()  # Release everything if job is finished

# %%
# create an animation with matplotlib with the frames, without axes and box
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.axis("off")  # Hide the axes
im = ax.imshow(
    np.zeros((500, 1000))
)  # Assuming frames are 640x480, adjust if different


def update(frame):
    im.set_data(frame)
    return [
        im,
    ]


ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

ani.save("belief_states_movie_zor.mp4", writer="ffmpeg", fps=10)
# %%
# redo the previous cell with moviepy, ensuring fps is a real number
from moviepy.editor import ImageSequenceClip

belief_state_images = [
    f"./figures/halvkdvk/{img}"
    for img in os.listdir("./figures/halvkdvk")
    if img.startswith("belief_states_model_epoch_") and img.endswith(".png") and not img.endswith("_datashader.png")
]

# only take images that have numbers that are multiples of 5
bfi = []
nums_ = []
for b in belief_state_images:
    # extract the numbers for the string
    nums = int(b.split("_")[-1].split(".")[0])
    if nums % 5 == 0:
        bfi.append(b)
        nums_.append(nums)

# now sort these according to number
bfi = [x for _, x in sorted(zip(nums_, bfi))]


belief_state_images = bfi

print(f"Number of images: {len(belief_state_images)}")

#%%
# Ensure fps is a real number, not NoneType or any other type
fps_value = 30.0

clip = ImageSequenceClip(belief_state_images, fps=fps_value)
clip.write_videofile("belief_states_movie_pretty.mp4", codec="libx264", audio=False)

# %%

# %%
# for illustrative purposes lets look at the last result
result = results[0]
fig = generate_belief_state_figures(
    result[2], result[3], result[4], plot_triangles=True, alpha=0.25
)
fig.show()
# %%
# lets get the logged data for the run
api = wandb.Api()
run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
run = api.run(run_path)
data = run.history(samples=95000)

# get columns that have val_relative_loss* in them
val_relative_loss_cols = [col for col in data.columns if "val_relative_loss" in col]
# get rid of nans
val_relative_loss = data[val_relative_loss_cols].dropna()
# order the columns
val_relative_loss = val_relative_loss[val_relative_loss.columns.sort_values()]

# set plotting style to talk
# %%
colors = plt.cm.viridis(np.linspace(0, 1, 10))
fig, ax = plt.subplots(
    figsize=(10, 6)
)  # Create a figure and an axes object for more control
current_iteration = 70050  # Example current iteration
inset_ax = fig.add_axes([0.47, 0.47, 0.5, 0.5])  # Corrected to tuple
ax.axvline(current_iteration, color="black", linestyle="--", linewidth=1, alpha=0.5)
inset_ax.axvline(
    current_iteration, color="black", linestyle="--", linewidth=1, alpha=0.5
)

for i, color in zip(range(10), colors):
    col_name = f"val_relative_loss_{i}"
    ax.plot(
        (val_relative_loss[col_name] - 1) * 100, color=color, linewidth=0.3, alpha=0.3
    )  # Convert fraction to percent above optimal
    smoothed_data = (
        val_relative_loss[col_name].rolling(window=100).mean() - 1
    ) * 100  # Convert fraction to percent above optimal
    ax.plot(smoothed_data, label=f"Position {str(i)}", color=color, linewidth=1)
    start_index = max(0, current_iteration - 15000)
    end_index = current_iteration + 15000
    inset_data = smoothed_data.loc[start_index:end_index]
    inset_ax.plot(inset_data, color=color, linewidth=1)

ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel(
    "Percent Above Optimal Loss", fontsize=20
)  # Change label to Percent Above Optimal
ax.tick_params(axis="both", which="major", labelsize=15)
ax.set_ylim(-0.01, 2)  # Adjust ylim to percent above optimal
plt.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
inset_ax.spines["top"].set_visible(False)
inset_ax.spines["right"].set_visible(False)

plt.show()
# %%
