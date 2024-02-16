#%%
import wandb

# Initialize wandb
wandb.init()

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
# Specify the project and run ID
api = wandb.Api()
user_or_org = 'adamimos'
project_name = 'transformer-MSPs'
run_id = '2zulyhrv'

# Access the run
run_path = f"{user_or_org}/{project_name}/{run_id}"
run = api.run(run_path)

artifacts = run.logged_artifacts()

# Optionally print the artifacts to find the one you're interested in
for artifact in artifacts:
    print(f"Artifact Name: {artifact.name}, Type: {artifact.type}, Version: {artifact.version}")

# Assuming you know the artifact name and type, you can directly use it
artifact_name = "model_epoch_149"  # This should match the exact name of your artifact
artifact_type = "model"  # This should match the type of your artifact
artifact_version = "v6"  # Optionally specify the version, if needed

# Construct the artifact reference
artifact_reference = f"{user_or_org}/{project_name}/{artifact_name}:{artifact_version}"

# Use the artifact
artifact = wandb.use_artifact(artifact_reference, type=artifact_type)
artifact_dir = artifact.download()

# Your artifact files are now downloaded to `artifact_dir`
print(f"Artifact downloaded to {artifact_dir}")
artifact_file = artifact_dir + "/model_epoch_149.pt"

# load artifiact, its a .pt file
model = torch.load(artifact_file, map_location=torch.device('cpu'))

df = run.history(samples=100000)
print(df.keys())
#%%
# Separate columns with 'val' in the name from those without
val_columns = [col for col in df.columns if 'val' in col]
non_val_columns = [col for col in df.columns if 'val' not in col]

# Display the separated columns
print("Columns with 'val':", val_columns)
print("Columns without 'val':", non_val_columns)

val_df = df[val_columns]
test_df = df[non_val_columns]

# delete nan rows in val_df
val_df = val_df.dropna()

# there are columns of the form val_relative_loss_N where N goes from 0
# to 9, lets take those values and make them a numpy array
val_relative_loss = val_df.filter(regex='val_relative_loss_*')
# make sure the columns are in order
val_relative_loss = val_relative_loss.sort_index(axis=1)
# convert to numpy array
val_relative_loss = val_relative_loss.to_numpy()
print(val_relative_loss.shape)

# make a plot of val_loss
import matplotlib.pyplot as plt

# make plots of val_relative_loss_N
import numpy as np
import matplotlib.cm as cm

colors = cm.viridis(np.linspace(0, 1, val_relative_loss.shape[1]))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for i in range(val_relative_loss.shape[1]):
    loss_ratio = val_relative_loss[:,i] / val_relative_loss[0,i]
    axs[0].plot(val_relative_loss[:,i], label=f'position {i}', color=colors[i])
    # Normalize so the first point is 1 and the last point is 0
    normalized_loss = (loss_ratio - loss_ratio[-1]) / (loss_ratio[0] - loss_ratio[-1])
    axs[1].plot(normalized_loss, label=f'position {i}', color=colors[i])

axs[0].set_title('Non-normalized Loss')
axs[1].set_title('Normalized Loss')
axs[0].legend()
axs[1].legend()
config = run.config
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
device = torch.device("cpu")
model = build_network(config, device)
model.load_state_dict(torch.load(artifact_file, map_location=device))



process = mess3(.05, .85)


print(config["n_ctx"])
MSP_tree = mixed_state_tree(process, config["n_ctx"] + 1)
myopic_entropy_rate = myopic_entropy(MSP_tree)
minimum_cross_entropy = myopic_entropy_rate[1:]
print(f"myopic_entropy_rate: {myopic_entropy_rate}")



minimum_cross_entropy = torch.tensor(minimum_cross_entropy, dtype=torch.float32).to(
    device
)

X_val, Y_val, val_weights = create_validation_set(MSP_tree, config['n_ctx'])
X_val = torch.tensor(X_val, dtype=torch.int).to(device)
val_weights = torch.tensor(val_weights, dtype=torch.float32).to(device)
Y_val = torch.tensor(Y_val, dtype=torch.long).to(device) # [batch_size, n_ctx]


# run the model on the validation set
model.eval()
batch_size = 100
all_losses = []
with torch.no_grad():
    for i in range(0, X_val.size(0), batch_size):
        X_val_batch = X_val[i:i+batch_size]
        Y_val_batch = Y_val[i:i+batch_size]
        val_weights_batch = val_weights[i:i+batch_size]
        
        logits = model(X_val_batch)  # [batch_size, n_ctx, d_vocab]
        # Flatten logits and Y_val for compatibility with F.cross_entropy
        logits_flat = logits.view(-1, logits.shape[-1])  # [batch_size * n_ctx, d_vocab]
        Y_val_flat = Y_val_batch.view(-1)  # [batch_size * n_ctx]
        # Calculate loss without reduction
        loss = F.cross_entropy(logits_flat, Y_val_flat, reduction='none')  # [batch_size * n_ctx]
        # Reshape loss to original [batch_size, n_ctx] to multiply by val_weights
        loss = loss.view(X_val_batch.shape[0], X_val_batch.shape[1]) # [batch_size, n_ctx]
        # Apply weights
        val_weights_repeat = val_weights_batch.unsqueeze(1).repeat(1, X_val_batch.shape[1])
        weighted_loss = loss * val_weights_repeat
        all_losses.append(weighted_loss)
        
# Concatenate all batch losses
all_losses = torch.cat(all_losses, dim=0)
# No mean calculation here, keeping nonreduced loss matrix
print(f'All losses shape: {all_losses.shape}')
#%%
# Calculate the mean loss for each position
mean_loss = all_losses.sum(dim=0)
print(f'Mean loss shape: {mean_loss.shape}')
# divide by myopic_entropy_rate to get the relative loss
relative_loss = mean_loss / minimum_cross_entropy
plt.plot(relative_loss)

#%%
# now i want to run all the data through the transformer using the 
# run_with_hooks function
from tqdm import tqdm
resid_acts = []
for i in tqdm(range(0, X_val.size(0), batch_size)):
    X_val_batch = X_val[i:i+batch_size]
    Y_val_batch = Y_val[i:i+batch_size]
    val_weights_batch = val_weights[i:i+batch_size]
    _, acts = model.run_with_cache(X_val_batch)
    acts = acts['ln_final.hook_normalized'] # [batch_size, n_ctx, d_model]
    resid_acts.append(acts)

resid_acts = torch.cat(resid_acts, dim=0) # [n_samples, n_ctx, d_model]
print(resid_acts.shape)
# %%
from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
    collect_path_probs_with_paths,
    collect_paths_with_beliefs
)
# for each position in each input dataset, we want to calculate the
# MSP state it corresponds to
path_probs = collect_path_probs_with_paths(MSP_tree, config['n_ctx'])
belief_states = MSP_tree.get_belief_states()
belief_states = np.array(belief_states)
print(belief_states.shape)
# plot belief_stest as a scatter plot
fig, ax = plt.subplots()
ax.scatter(belief_states[:,0], belief_states[:,1])
ax.set_xlabel('belief state 0')
ax.set_ylabel('belief state 1')

#%%
# run linear regression from the activations to the belief states
results = collect_paths_with_beliefs(MSP_tree, config['n_ctx'])
seqs = [x[0] for x in results]
beliefs = [x[2] for x in results]


indices = []
seqs_beliefs = []
for x in tqdm(X_val):
    # convert to tuple
    x = tuple(x.tolist())
    # find the index of x in seqs
    idx = seqs.index(x)
    seqs_beliefs.append(beliefs[idx])
#%%
all_acts = []
all_beliefs = []
for i in range(config['n_ctx']):
    print(i)
    acts = resid_acts[:,i,:] # [n_samples, d_model]
    # to numpy
    acts = acts.detach().cpu().numpy()
    results = collect_paths_with_beliefs(MSP_tree, i+1)
    seqs = [s[0] for s in results]
    beliefs = [s[2] for s in results]
    seqs_beliefs = []
    for x in tqdm(X_val[:,:i+1]):
        # convert to tuple
        x = x.tolist()
        if type(x) == list:
            x = tuple(x)
        else:
            x = (x,)

        # find the index of x in seqs
        idx = seqs.index(x)
        seqs_beliefs.append(beliefs[idx])

    seqs_beliefs = np.array(seqs_beliefs)
    print(seqs_beliefs.shape) # [n_samples, 3]
    print(acts.shape) # [n_samples, 64]
    all_acts.append(acts)
    all_beliefs.append(seqs_beliefs)

all_acts = np.concatenate(all_acts, axis=0)
all_beliefs = np.concatenate(all_beliefs, axis=0)
print(all_acts.shape)
print(all_beliefs.shape)


# %%
# run linear regression from the activations to the belief states
# this is multioutput regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(all_acts, all_beliefs)
# %%
# now we can use the regression to predict the belief states from the activations
# and then plot the belief states
predicted_beliefs = reg.predict(all_acts)
print(predicted_beliefs.shape)

# %%

def project_to_simplex(points):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    # Assuming points is a 2D array with shape (n_points, 3)
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y


fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Ground Truth
colors_gt = belief_states
x_gt, y_gt = project_to_simplex(np.array(belief_states))
axs[0].scatter(x_gt, y_gt, c=colors_gt, s=0.05)  # Made points smaller
axs[0].set_title('Ground Truth')
axs[0].axis('off')
# plot the equilateral triangle
#axs[0].plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-')

# Residual Stream
colors_rs = all_beliefs
x_rs, y_rs = project_to_simplex(np.array(predicted_beliefs))
axs[1].scatter(x_rs, y_rs, c=colors_rs, s=0.05)  # Made points smaller
axs[1].set_title('Residual Stream')
axs[1].axis('off')
# plot the equilateral triangle
axs[1].plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-')
# zoom into 0-0.4 in each dimension


plt.show()
# %%
# now lets do a shuffle control where we shuffle the labels
# and then run the regression again
all_beliefs_shuffled = np.random.permutation(all_beliefs)
reg_shuffle = LinearRegression().fit(all_acts, all_beliefs_shuffled)
predicted_beliefs_shuffle = reg_shuffle.predict(all_acts)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Ground Truth
colors_gt = belief_states
x_gt, y_gt = project_to_simplex(np.array(belief_states))
axs[0].scatter(x_gt, y_gt, c=colors_gt, s=0.05)  # Made points smaller
axs[0].set_title('Ground Truth')
axs[0].axis('off')
# plot the equilateral triangle
#axs[0].plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-')

# Residual Stream
colors_rs = all_beliefs_shuffled
x_rs, y_rs = project_to_simplex(np.array(predicted_beliefs_shuffle))
axs[1].scatter(x_rs, y_rs, c=colors_rs, s=0.05)  # Made points smaller
axs[1].set_title('Residual Stream Shuffled')
axs[1].axis('off')
#axs[1].plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'k-')

# %%
