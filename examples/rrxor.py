#%%
from epsilon_transformers.process.processes import RRXOR
import numpy as np
from epsilon_transformers.training.configs import RawModelConfig
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import mean_squared_error

def load_model(config, device, download_dir, save_point):
    model = model_config = RawModelConfig(
        d_vocab=config["d_vocab"],
        d_model=config["d_model"],
        n_ctx=config["n_ctx"],
        d_head=config["d_head"],
        n_head=config["n_heads"],
        d_mlp=config["d_model"] * 4,
        n_layers=config["n_layers"],
    ).to_hooked_transformer(seed=1337, device=device)

    model.load_state_dict(torch.load(f"{download_dir}/model_epoch_{save_point}.pt", map_location=device))
    return model

def process_msp(config):
    process = RRXOR()
    msp = process.derive_mixed_state_presentation(depth=config["n_ctx"] + 1)
    msp_paths_and_beliefs = msp.paths_and_belief_states

    msp_paths = [np.array(x[0]) for x in msp_paths_and_beliefs]
    msp_beliefs = [x[1] for x in msp_paths_and_beliefs]
    msp_beliefs = [tuple(round(b, 5) for b in belief) for belief in msp_beliefs]
    msp_unique_beliefs = list(set(msp_beliefs))  # Convert to list for indexing
    print(f"Unique beliefs: {len(msp_unique_beliefs)} out of {len(msp_beliefs)}")

    return msp, msp_paths, msp_beliefs, msp_unique_beliefs

def perform_pca(msp_unique_beliefs):
    pca = PCA(n_components=3)
    pca.fit(msp_unique_beliefs)
    msp_beliefs_pca = pca.transform(msp_unique_beliefs)
    return pca, msp_beliefs_pca

def prepare_data(msp, msp_paths, msp_beliefs, msp_unique_beliefs, config, device):
    msp_belief_index = {b: i for i, b in enumerate(msp_unique_beliefs)}
    msp_beliefs_index = [msp_belief_index[tuple(round(b, 5) for b in belief)] for belief in msp_unique_beliefs]

    X = [x for x in msp.paths if len(x) == config["n_ctx"]]
    X = torch.tensor(X, dtype=torch.int).to(device)  # (batch, n_ctx)

    X_beliefs = torch.zeros(X.shape[0], X.shape[1], 5).to(device)
    X_belief_indices = torch.zeros(X.shape[0], X.shape[1], 1, dtype=torch.long).to(device)  # Specify dtype

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = X[i, :j+1].cpu().numpy()
            path_index = np.where([np.array_equal(p, x) for x in msp_paths])[0][0]  # Get the first index
            msp_belief_state = msp_beliefs[path_index]
            X_beliefs[i, j] = torch.tensor(msp_belief_state, dtype=torch.float32)
            X_belief_indices[i, j] = msp_belief_index[msp_belief_state]  # Assign directly

    return X, X_beliefs, X_belief_indices, msp_beliefs_index

def get_activations(model, X):
    _, cache = model.run_with_cache(X, names_filter=lambda x: 'ln1.hook_normalized' in x)
    acts = torch.cat([cache[f"blocks.{i}.ln1.hook_normalized"] for i in range(4)], dim=-1)  # (batch, n_ctx, 4 * d_model)
    return acts


def perform_regression(acts, X_beliefs, X_belief_indices):
    acts_flattened = acts.view(-1, acts.shape[-1]).cpu().numpy()
    X_beliefs_flattened = X_beliefs.view(-1, X_beliefs.shape[-1]).cpu().numpy()
    X_belief_indices_flattened = X_belief_indices.view(-1).cpu().numpy()

    regression = LinearRegression()
    regression.fit(acts_flattened, X_beliefs_flattened)
    result = regression.predict(acts_flattened)

    belief_errors = {}
    for b in range(len(msp_unique_beliefs)):
        relevant_indices = np.where(X_belief_indices_flattened == b)[0]
        if relevant_indices.size > 0:
            errors = np.abs(X_beliefs_flattened[relevant_indices] - result[relevant_indices])
            mean_error = np.mean(errors)
            belief_errors[b] = mean_error
        else:
            belief_errors[b] = np.nan

    return result, belief_errors

def visualize_results(msp_beliefs_pca, msp_beliefs_index, result, pca, X_belief_indices):
    colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

    msp_beliefs_pca_data = msp_beliefs_pca
    for b in range(len(msp_unique_beliefs)):
        relevant_indices = [i for i, x in enumerate(msp_beliefs_index) if x == b]  # Use list comprehension
        if not relevant_indices:
            print(f"No relevant indices found for belief {b}. Skipping this belief.")
            continue
        relevant_data = msp_beliefs_pca_data[relevant_indices]
        fig.add_trace(go.Scatter3d(x=relevant_data[:, 0],
                                   y=relevant_data[:, 1],
                                   z=relevant_data[:, 2],
                                   mode='markers',
                                   name=f'Belief {b}',
                                   marker=dict(size=5, color=colors[b], opacity=1.0)),
                      row=1, col=1)

    result_pca = pca.transform(result)
    X_belief_indices_flattened = X_belief_indices.view(-1).cpu().numpy()

    for b in range(len(msp_unique_beliefs)):
        relevant_indices = np.where(X_belief_indices_flattened == b)[0]
        if not relevant_indices.size:
            print(f"No relevant indices found for belief {b} in the transformed results. Skipping this belief.")
            continue
        relevant_data = result_pca[relevant_indices]
        if relevant_data.size == 0:
            print(f"No relevant data found for belief {b} after PCA transformation. Skipping this belief.")
            continue
        centers_of_mass = np.mean(relevant_data, axis=0)
        fig.add_trace(go.Scatter3d(x=[centers_of_mass[0]],
                                   y=[centers_of_mass[1]],
                                   z=[centers_of_mass[2]],
                                   mode='markers',
                                   name=f'Belief {b}',
                                   marker=dict(size=5, color=colors[b], opacity=1)),
                      row=1, col=2)
        fig.add_trace(go.Scatter3d(x=relevant_data[:, 0],
                                   y=relevant_data[:, 1],
                                   z=relevant_data[:, 2],
                                   mode='markers',
                                   name=f'Belief {b}',
                                   marker=dict(size=1.5, color=colors[b], opacity=0.1)),
                      row=1, col=2)

    fig.update_layout(title='3D PCA Projection of Beliefs',
                      scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                      scene2=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))
    fig.show()


def analyze_regression_error(save_point_numbers, config, device, download_dir, X, X_beliefs, X_belief_indices, msp_beliefs_index):
    regression_errors = {}

    for save_point in tqdm(save_point_numbers[::20]):
        model = load_model(config, device, download_dir, save_point)
        acts = get_activations(model, X)
        _, belief_errors = perform_regression(acts, X_beliefs, X_belief_indices)
        regression_errors[save_point] = belief_errors

    return regression_errors

def plot_regression_errors(regression_errors):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for b in range(len(msp_unique_beliefs)):
        errors = [regression_errors[save_point][b] for save_point in save_point_numbers]
        ax.plot(save_point_numbers, errors, marker='o', label=f'Belief {b}')
    
    ax.set_xlabel('Save Point')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Regression Error over Save Points')
    ax.legend()
    
    plt.show()

#%%
import torch
import wandb
from epsilon_transformers.analysis.wandb import fetch_artifacts_for_run
from epsilon_transformers.analysis.wandb import fetch_run_config
from epsilon_transformers.analysis.wandb import load_model_artifact
from epsilon_transformers.analysis.wandb import download_artifacts


if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("mps")
print(f"Using device: {device}")

user_or_org = "adamimos"
project_name = "transformer-MSPs"
run_id = "vfs4q106"  # rrxor 
#%%
wandb.init(id=run_id, resume='must')
arts = fetch_artifacts_for_run(user_or_org, project_name, run_id)
print(f"Found {len(arts)} artifacts")


download_dir = "./downloaded_artifacts"  # Specify your directory here
# download_artifacts(arts, download_dir)

config = fetch_run_config(user_or_org, project_name, run_id)
#%%
import os
# get the number in each filename in the download_dir
# and make it a list of ints, then sort it and see if any are missing
filenames = [int(x.split("_")[-1].split(".")[0]) for x in os.listdir(download_dir)]
filenames.sort()
print(f"Missing: {set(range(filenames[0], filenames[-1])) - set(filenames)}")

save_point = 10
model = load_model(config, device, download_dir, save_point)
msp, msp_paths, msp_beliefs, msp_unique_beliefs = process_msp(config)
pca, msp_beliefs_pca = perform_pca(msp_unique_beliefs)
X, X_beliefs, X_belief_indices, msp_beliefs_index = prepare_data(msp, msp_paths, msp_beliefs, msp_unique_beliefs, config, device)
acts = get_activations(model, X)
result = perform_regression(acts, X_beliefs)
visualize_results(msp_beliefs_pca, msp_beliefs_index, result, pca, X_belief_indices)

# %%
# get list of all save_points
import glob
save_points = glob.glob(f"{download_dir}/*.pt")
save_point_numbers = [int(x.split("_")[-1].split(".")[0]) for x in save_points]
save_point_numbers.sort()
from tqdm import tqdm


for save_point in tqdm(save_point_numbers):
    model = load_model(config, device, download_dir, save_point)
    acts = get_activations(model, X)
    result = perform_regression(acts, X_beliefs)
    visualize_results(msp_beliefs_pca, msp_beliefs_index, result, pca, X_belief_indices)   
# %%
regression_errors = analyze_regression_error(save_point_numbers, config, device, download_dir, X, X_beliefs, X_belief_indices, msp_beliefs_index)
#%%
# plot_regression_errors(regression_errors)
from matplotlib import pyplot as plt
data = []
for epoch in regression_errors.keys():
    data.append(list(regression_errors[epoch].values()))

data = np.array(data) # (n_epochs, n_beliefs)

# take out the belief that has nans
data = data[:, ~np.isnan(data).any(axis=0)]
data_normalized = data 

# Plotting with plotly using log scale for both axes
import plotly.graph_objects as go

fig = go.Figure()
for i in range(data_normalized.shape[1]):
    data_ = data_normalized[:, i]
    data_smoothed = np.convolve(data_, np.ones(10)/10, mode='valid')
    fig.add_trace(go.Scatter(x=np.arange(1, len(data_smoothed)+1), y=data_smoothed, mode='lines', name=f'Belief {i+1}'))

fig.update_layout(title='Normalized Data Visualization', xaxis_title='Epoch', yaxis_title='Normalized Value')
fig.show()

# %%
