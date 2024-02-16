# type: ignore
# %%
import torch
import wandb
import numpy as np
from epsilon_transformers import build_network
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#%%
import wandb


from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
    collect_path_probs_with_paths,
    collect_paths_with_beliefs
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
#%%

def load_model_artifact(user_or_org, project_name, artifact_name, artifact_type, artifact_version, config, device='cpu'):
    api = wandb.Api()
    artifact_reference = f"{user_or_org}/{project_name}/{artifact_name}:{artifact_version}"
    print(f"Loading artifact {artifact_reference}")
    artifact = wandb.use_artifact(artifact_reference, type=artifact_type)
    artifact_dir = artifact.download()
    artifact_file = f"{artifact_dir}/{artifact_name}.pt"  # Making the filename programmatic
    model = build_network(config, torch.device(device))
    model.load_state_dict(torch.load(artifact_file, map_location=device))
    return model

def prepare_data(config, device='cpu'):
    # Assume MSP_tree and process are defined globally or passed as parameters
    X_val, Y_val, val_weights = create_validation_set(MSP_tree, config['n_ctx'])
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
            X_val_batch = X_val[i:i+batch_size]
            Y_val_batch = Y_val[i:i+batch_size]
            val_weights_batch = val_weights[i:i+batch_size]
            logits = model(X_val_batch)
            logits_flat = logits.view(-1, logits.shape[-1])
            Y_val_flat = Y_val_batch.view(-1)
            loss = F.cross_entropy(logits_flat, Y_val_flat, reduction='none')
            loss = loss.view(X_val_batch.shape[0], X_val_batch.shape[1])
            weighted_loss = loss * val_weights_batch.unsqueeze(1).repeat(1, X_val_batch.shape[1])
            all_losses.append(weighted_loss)
    return torch.cat(all_losses, dim=0)

def run_linear_regression(activations, beliefs):
    reg = LinearRegression().fit(activations, beliefs)
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
            X_val_batch = X_val[i:i+batch_size].to(device)
            # Utilize the model's caching mechanism during forward pass
            _, acts = model.run_with_cache(X_val_batch)
            # Assuming 'ln_final.hook_normalized' is the key for the activations we're interested in
            acts = acts['ln_final.hook_normalized']  # [batch_size, n_ctx, d_model]
            activations.append(acts.cpu())
    
    activations = torch.cat(activations, dim=0)  # Concatenate all batch activations
    return activations.numpy()  # Convert to numpy array for consistency with the rest of the codebase


def precompute_activation_and_belief_inds(MSP_tree, n_ctx, X_val):
    """
    X_val: [n_samples, n_ctx]
    n_ctx: int
    """
    all_beliefs = np.zeros((X_val.size(0), n_ctx, 3))

    for i in tqdm(range(n_ctx)):
        results = collect_paths_with_beliefs(MSP_tree, i+1)
        # Convert sequences and beliefs into a dictionary for faster lookup
        seqs_to_beliefs = {tuple(s[0]): s[2] for s in results}
        # Pre-convert all X_val sequences up to i+1 into tuples for faster batch processing
        x_tuples = [tuple(x[:i+1].tolist()) for x in X_val]
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
    for i in range(config['n_ctx']):
        acts = activations[:,i,:] # [n_samples, d_model]
        results = collect_paths_with_beliefs(MSP_tree, i+1)
        seqs = [s[0] for s in results]
        beliefs = [s[2] for s in results]
        seqs_beliefs = []
        for x in tqdm(X_val[:,:i+1]):
            # convert to tuple
            x = tuple(x.tolist()) if type(x.tolist()) == list else (x.tolist(),)
            # find the index of x in seqs
            idx = seqs.index(x)
            seqs_beliefs.append(beliefs[idx])
        seqs_beliefs = np.array(seqs_beliefs)
        all_acts.append(acts)
        all_beliefs.append(seqs_beliefs)

    all_acts = np.concatenate(all_acts, axis=0) # [n_samples * n_ctx, d_model]
    all_beliefs = np.concatenate(all_beliefs, axis=0) # [n_samples * n_ctx, 3]
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
def generate_belief_state_figures_datashader(belief_states, all_beliefs, predicted_beliefs, plot_triangles=False):
    # Projection and DataFrame preparation
    bs_x, bs_y = project_to_simplex(np.array(belief_states))
    df_gt = pd.DataFrame({'x': bs_x, 'y': bs_y, 'r': belief_states[:, 0], 'g': belief_states[:, 1], 'b': belief_states[:, 2]})

    pb_x, pb_y = project_to_simplex(np.array(predicted_beliefs))
    df_pb = pd.DataFrame({'x': pb_x, 'y': pb_y, 'r': all_beliefs[:, 0], 'g': all_beliefs[:, 1], 'b': all_beliefs[:, 2]})

    # Create canvas
    cvs = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(-0.1, 1.1), y_range=(-0.1, np.sqrt(3)/2 + 0.1))
    # Aggregate each RGB channel separately for ground truth and predicted beliefs
    agg_funcs = {'r': ds.mean('r'), 'g': ds.mean('g'), 'b': ds.mean('b')}
    agg_gt = {color: cvs.points(df_gt, 'x', 'y', agg_funcs[color]) for color in ['r', 'g', 'b']}
    agg_pb = {color: cvs.points(df_pb, 'x', 'y', agg_funcs[color]) for color in ['r', 'g', 'b']}

    # Combine aggregated channels into RGB images
    def combine_channels_to_rgb(agg_r, agg_g, agg_b):
        img_r = tf.shade(agg_r, cmap=['black', 'red'], how='linear')
        img_g = tf.shade(agg_g, cmap=['black', 'green'], how='linear')
        img_b = tf.shade(agg_b, cmap=['black', 'blue'], how='linear')

        img_r = tf.spread(img_r, px=1, shape='circle')
        img_g = tf.spread(img_g, px=1, shape='circle')
        img_b = tf.spread(img_b, px=1, shape='circle')

        # Combine using numpy
        r_array = np.array(img_r.to_pil()).astype(np.float64)
        g_array = np.array(img_g.to_pil()).astype(np.float64)
        b_array = np.array(img_b.to_pil()).astype(np.float64)
        
        # Stack arrays into an RGB image (ignoring alpha channel for simplicity)
        rgb_image = np.stack([r_array[:,:,0], g_array[:,:,1], b_array[:,:,2]], axis=-1)
        

        
        return Image.fromarray(np.uint8(rgb_image))

    img_gt = combine_channels_to_rgb(agg_gt['r'], agg_gt['g'], agg_gt['b'])
    img_pb = combine_channels_to_rgb(agg_pb['r'], agg_pb['g'], agg_pb['b'])



    # Visualization with Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, facecolor='black')  # Changed 'white' to 'black'
    for ax in axs:
        ax.tick_params(axis='x', colors='black')  # Changed 'black' to 'white'
        ax.tick_params(axis='y', colors='black')  # Changed 'black' to 'white'
        ax.xaxis.label.set_color('black')  # Changed 'black' to 'white'
        ax.yaxis.label.set_color('black')  # Changed 'black' to 'white'
        ax.title.set_color('black')  # Changed 'black' to 'white'
    axs[0].imshow(img_gt)
    axs[1].imshow(img_pb)
    
    axs[0].axis('off')
    axs[1].axis('off')
    title_y_position = -0.1  # Adjust this value to move the title up or down relative to the axes
    fig.text(0.5, title_y_position, 'Ground Truth', ha='center', va='top', transform=axs[0].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'
    fig.text(0.5, title_y_position, 'Residual Stream', ha='center', va='top', transform=axs[1].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'

        
    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'white', lw=2)  # Changed 'black' to 'white'

    return fig


def generate_belief_state_figures(belief_states, all_beliefs, predicted_beliefs, plot_triangles=False,
                                  alpha=0.5, s=0.01, show_plot=False):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # Ground Truth
    colors_gt = belief_states
    x_gt, y_gt = project_to_simplex(np.array(belief_states))
    
    # randomly permute the rows
    axs[0].scatter(x_gt, y_gt, c=colors_gt, s=s, alpha=1.)
    axs[0].set_title('Ground Truth', pad=-20)
    axs[0].axis('off')

    # Residual Stream
    colors_rs = all_beliefs # [n_samples * n_ctx, 3]
    x_rs, y_rs = project_to_simplex(np.array(predicted_beliefs))
    axs[1].scatter(x_rs, y_rs, c=colors_rs, s=s, alpha=alpha)
    axs[1].set_title('Residual Stream', pad=-20)
    axs[1].axis('off')

    # Plot the equilateral triangle if requested
    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-')

    # set x y limits
    axs[0].set_xlim(-0.1, 1.1)
    axs[0].set_ylim(-0.1, 1.1)
    # Close the plot if not intended to show
    if not show_plot:
        plt.close(fig)
    # Return the figure object for further manipulation or saving
    return fig
#%%


device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"Using device: {device}")
wandb.init()
user_or_org = 'adamimos'
project_name = 'transformer-MSPs'
#run_id = '2zulyhrv' # mess3 param change
#run_id = 's6p0aaci' # zero one random
run_id = 'halvkdvk' # mess3 param change long run

#%%

def fetch_artifacts_for_run(user_or_org, project_name, run_id):
    api = wandb.Api()
    run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
    run = api.run(run_path)
    artifacts = run.logged_artifacts()
    return artifacts

arts = fetch_artifacts_for_run(user_or_org, project_name, run_id)
#%%
config = fetch_run_config(user_or_org, project_name, run_id)
process = mess3(.05, .85)
#process = zero_one_random()
MSP_tree = mixed_state_tree(process, config["n_ctx"] + 1)
X_val, Y_val, val_weights = create_validation_set(MSP_tree, config['n_ctx'])

X_val = torch.tensor(X_val, dtype=torch.int).to(device)
val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)

all_beliefs = precompute_activation_and_belief_inds(MSP_tree, config['n_ctx'], X_val)
# all_beliefs is shape [n_samples, n_ctx, 3]
#%%
results = []
for art in tqdm(arts):
    art_name = art.name
    print(f"Artifact: {art_name}")
    # art.name is "name:version" so we split it to get the name and version
    artifact_name, artifact_version = art_name.split(":")
    def analyze_artifact(user_or_org, project_name, artifact_name, artifact_type, artifact_version, config, device, all_beliefs):
        model = load_model_artifact(user_or_org, project_name, artifact_name, artifact_type, artifact_version, config, device=device)

        #myopic_entropy_rate = myopic_entropy(MSP_tree)
        #minimum_cross_entropy = myopic_entropy_rate[1:]
        path_probs = collect_path_probs_with_paths(MSP_tree, config['n_ctx'])
        belief_states = MSP_tree.get_belief_states()
        belief_states = np.array(belief_states)
        
        activations = extract_activations_with_cache(model, X_val, device, batch_size=10000)
        # activations is shape [n_samples, n_ctx, d_model]
        # reshape all_beliefs and activations
        all_beliefs_reshaped = all_beliefs.reshape(-1, all_beliefs.shape[-1])
        all_acts = activations.reshape(-1, activations.shape[-1])
        reg, predicted_beliefs = run_linear_regression(all_acts, all_beliefs_reshaped)
        
        # add the image generation here

        #fig = generate_belief_state_figures(belief_states, all_beliefs_reshaped, predicted_beliefs, plot_triangles=True)
        fig = generate_belief_state_figures_datashader(belief_states, all_beliefs_reshaped, predicted_beliefs, plot_triangles=True)
        # belief_states # [n_ctx, 3], all_beliefs_reshaped # [n_samples * n_ctx, 3], predicted_beliefs # [n_samples * n_ctx, 3]
        fig.savefig(f'figures/belief_states_{artifact_name}.png', dpi=300)
        return artifact_name, reg, belief_states, all_beliefs_reshaped, predicted_beliefs
    #project_name = 'zero_one_random_initial_sweep'
    result = analyze_artifact(user_or_org, project_name, artifact_name, art.type, art.version, config, device, all_beliefs)
    # move results to cpu
    #results.append(result)
    # plot_belief_states(result[2], result[3], result[4])

# %%
# save the results to file
import pickle

with open('results_z1r.pkl', 'wb') as f:
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
ani.save('belief_states_animation.mp4', writer='ffmpeg', fps=30)


#%%
# load results
import pickle
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
# %%
for i, r in tqdm(enumerate(results)):
    if i >= -1:
        fig = generate_belief_state_figures(r[2], r[3], r[4], plot_triangles=True, alpha=1.0, s=15)
        # save the figure
        fig.savefig(f'belief_states_zor_{i}.png')


# %%
import cv2
import os

# Define the codec and create VideoWriter object for AVI format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('belief_states_movie.avi', fourcc, 10.0, (640, 480))

belief_state_images = [img for img in os.listdir() if img.startswith('belief_states_z') and img.endswith('.png')]
belief_state_images.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))  # Sort files by the number in their name
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
ax.axis('off')  # Hide the axes
im = ax.imshow(np.zeros((500, 1000)))  # Assuming frames are 640x480, adjust if different

def update(frame):
    im.set_data(frame)
    return [im,]

ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=True)

ani.save('belief_states_movie_zor.mp4', writer='ffmpeg', fps=10)
# %%
# redo the previous cell with moviepy, ensuring fps is a real number
from moviepy.editor import ImageSequenceClip

belief_state_images = [img for img in os.listdir() if img.startswith('belief_states_') and img.endswith('.png')]
belief_state_images.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))  # Sort files by the number in their name

# Ensure fps is a real number, not NoneType or any other type
fps_value = 30.0

clip = ImageSequenceClip(belief_state_images, fps=fps_value)
clip.write_videofile("belief_states_movie.mp4", codec="libx264", audio=False)

# %%

# %%
# for illustrative purposes lets look at the last result
result = results[0]
fig = generate_belief_state_figures(result[2], result[3], result[4],
                                    plot_triangles=True, alpha=0.25)
fig.show()
# %%
# lets get the logged data for the run
api = wandb.Api()
run_path = f"{user_or_org}/{project_name}/runs/{run_id}"
run = api.run(run_path)
data = run.history(samples=95000)

# get columns that have val_relative_loss* in them
val_relative_loss_cols = [col for col in data.columns if 'val_relative_loss' in col]
# get rid of nans
val_relative_loss = data[val_relative_loss_cols].dropna()
# order the columns
val_relative_loss = val_relative_loss[val_relative_loss.columns.sort_values()]

# set plotting style to talk
#%%
colors = plt.cm.viridis(np.linspace(0, 1, 10))
fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and an axes object for more control
current_iteration = 70050  # Example current iteration
inset_ax = fig.add_axes([0.47, 0.47, 0.5, 0.5])  # Corrected to tuple
ax.axvline(current_iteration, color='black', linestyle='--', linewidth=1, alpha=0.5)
inset_ax.axvline(current_iteration, color='black', linestyle='--', linewidth=1, alpha=0.5)

for i, color in zip(range(10), colors):
    col_name = f'val_relative_loss_{i}'
    ax.plot((val_relative_loss[col_name] - 1) * 100, color=color, linewidth=.3, alpha=0.3)  # Convert fraction to percent above optimal
    smoothed_data = (val_relative_loss[col_name].rolling(window=100).mean() - 1) * 100  # Convert fraction to percent above optimal
    ax.plot(smoothed_data, label=f'Position {str(i)}', color=color, linewidth=1)
    start_index = max(0, current_iteration - 15000)
    end_index = current_iteration + 15000
    inset_data = smoothed_data.loc[start_index:end_index]
    inset_ax.plot(inset_data, color=color, linewidth=1)

ax.set_xlabel('Iteration', fontsize=20)
ax.set_ylabel('Percent Above Optimal Loss', fontsize=20)  # Change label to Percent Above Optimal
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_ylim(-0.01, 2)  # Adjust ylim to percent above optimal
plt.tight_layout()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
inset_ax.spines['top'].set_visible(False)
inset_ax.spines['right'].set_visible(False)

plt.show()
# %%
