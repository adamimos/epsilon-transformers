# %%

%load_ext autoreload
%autoreload 2

from epsilon_transformers.persistence import S3Persister
from epsilon_transformers.training.configs.model_configs import RawModelConfig
from epsilon_transformers.process.processes import RRXOR
from epsilon_transformers.analysis.activation_analysis import get_beliefs_for_transformer_inputs

import numpy as np
import torch
import plotly.express as px


from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

#%%
persister = S3Persister(collection_location="rrxor")

def get_model_checkpoints(persister: S3Persister):
    filenames = persister.list_objects()
    filenames_pt = [x for x in filenames if ".pt" in x]
    filenames_pt.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return filenames_pt

checkpoint_filenames = get_model_checkpoints(persister)
print(f"Checkpoints found: {len(checkpoint_filenames)}")
print(f"Last checkpoint: {checkpoint_filenames[-1]}")

train_config = persister.load_json(object_name='train_config.json')
print("\n".join(f"{k}: {v}" for k, v in train_config.items()))
#%%

process = RRXOR()
mixed_state_tree = process.derive_mixed_state_presentation(depth=train_config["n_ctx"] + 1)
# in order to plot the belief states in the simplex, we need to get the paths and beliefs from the MSP
tree_paths, tree_beliefs = mixed_state_tree.paths_and_belief_states

# lets print out the first few paths and beliefs
for path, belief in zip(tree_paths[:5], tree_beliefs[:5]):
    print(f"Path: {''.join([str(x) for x in path])}, Belief: {belief}")

# the MSP states are the unique beliefs in the tree
msp_beliefs = [tuple(round(b, 5) for b in belief) for belief in tree_beliefs]
print(f"Number of Unique beliefs: {len(set(msp_beliefs))} out of {len(msp_beliefs)}")

# now lets index each belief
msp_belief_index = {b: i for i, b in enumerate(set(msp_beliefs))}

for i in range(5):
    ith_belief = list(msp_belief_index.keys())[i]
    print(f"{ith_belief} is indexed as {msp_belief_index[ith_belief]}")

#%% FIRST VIZ

def run_visualization_pca(beliefs):
    pca = PCA(n_components=3)
    pca.fit(beliefs)

    return pca

def visualize_ground_truth_simplex_3d(beliefs, belief_labels, pca):

    beliefs_pca = pca.transform(beliefs)

    colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly
    fig = px.scatter_3d(beliefs_pca, x=0, y=1, z=2,
                        color=[str(i) for i in belief_labels],
                        color_discrete_sequence=colors)
    fig.show()

vis_pca = run_visualization_pca(list(msp_belief_index.keys()))
index = list(msp_belief_index.values())
visualize_ground_truth_simplex_3d(list(msp_belief_index.keys()),
                                  list(msp_belief_index.values()),
                                  vis_pca)


#%%

# now lets set up all the inputs as they arrive into the transformer
device = 'cpu'
transformer_inputs = [x for x in tree_paths if len(x) == train_config["n_ctx"]]
transformer_inputs = torch.tensor(transformer_inputs, dtype=torch.int).to(device)

# print first few batches
print(transformer_inputs[:5])

transformer_input_beliefs, transformer_input_belief_indices = get_beliefs_for_transformer_inputs(transformer_inputs, msp_belief_index, tree_paths, tree_beliefs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = persister.load_model(object_name=checkpoint_filenames[-1], device='cpu')

_, activations = model.run_with_cache(transformer_inputs, names_filter=lambda x: 'resid' in x)

print(activations.keys())


# we now have activations [batch, n_ctx, d_model]
# and we have transformer_input_beliefs [batch, n_ctx, belief_dim]
# and we have transformer_input_belief_indices [batch, n_ctx]

# in the end we want to do linear regression between the activations and the transformer_input_beliefs
def run_activation_to_beliefs_regression(activations, ground_truth_beliefs):

    # make sure the first two dimensions are the same
    assert activations.shape[0] == ground_truth_beliefs.shape[0]
    assert activations.shape[1] == ground_truth_beliefs.shape[1]

    # flatten the activations
    batch_size, n_ctx, d_model = activations.shape
    belief_dim = ground_truth_beliefs.shape[-1]
    activations_flattened = activations.view(-1, d_model) # [batch * n_ctx, d_model]
    ground_truth_beliefs_flattened = ground_truth_beliefs.view(-1, belief_dim) # [batch * n_ctx, belief_dim]
    
    # run the regression
    regression = LinearRegression()
    regression.fit(activations_flattened, ground_truth_beliefs_flattened)

    # get the belief predictions
    belief_predictions = regression.predict(activations_flattened) # [batch * n_ctx, belief_dim]
    belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)

    return regression, belief_predictions

#acts = torch.concatenate((activations["blocks.0.ln1.hook_normalized"], activations["blocks.1.ln1.hook_normalized"], activations["blocks.2.ln1.hook_normalized"], activations["blocks.3.ln1.hook_normalized"]), dim=-1)
#acts = activations["blocks.3.hook_resid_post"]
# torch concat everything in activations
acts = torch.concatenate(list(activations.values()), dim=-1)
print(acts.shape)
regression, belief_predictions = run_activation_to_beliefs_regression(acts, transformer_input_beliefs)
print(belief_predictions.shape)


belief_predictions_pca = vis_pca.transform(belief_predictions.reshape(-1, 5))
#%% SECOND VIZ
# now visualize in 3d
transformer_input_belief_indices_flattened = transformer_input_belief_indices.view(-1).cpu().numpy()
from plotly.subplots import make_subplots
from plotly import graph_objects as go

colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24 + px.colors.qualitative.Plotly

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])
for belief in msp_belief_index.keys():
    b = msp_belief_index[belief]
    relevant_indices = np.where(transformer_input_belief_indices_flattened == b)[0]
    relevant_data = belief_predictions_pca[relevant_indices]
    if len(relevant_data) > 0:
        centers_of_mass = np.mean(relevant_data, axis=0)

        fig.add_trace(go.Scatter3d(x=relevant_data[:, 0],
                                   y=relevant_data[:, 1],
                                   z=relevant_data[:, 2],
                                   mode='markers', name=f'Belief {b}', marker=dict(size=2, color=colors[b], opacity=.1)),
                                   row=1, col=2)
    
        fig.add_trace(go.Scatter3d(x=[centers_of_mass[0]],
                                y=[centers_of_mass[1]],
                                z=[centers_of_mass[2]],
                                mode='markers',
                                name=f'Belief {b}',
                                marker=dict(size=5, color=colors[b], opacity=1)),
                    row=1, col=2)
        
fig.update_layout(title='3D PCA Projection of Beliefs',
                    scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'),
                    scene2=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))
fig.show()



# %%
T = RRXOR().transition_matrix
msp_belief_token_prediction = {}
for b, i in msp_belief_index.items():
    b_numpy = np.array(b)
    optimal_output = np.einsum('esd,s->e', T, b_numpy)
    msp_belief_token_prediction[b] = optimal_output



def string_key(values, precision=5):
    return tuple(f"{round(value, precision):.{precision}f}" for value in values)

def create_string_key(values, precision=5):
    return tuple(f"{round(value, precision):.{precision}f}" for value in values)

# Example of setting up the dictionary with string keys
msp_belief_token_prediction_str = {}
# Populate the dictionary, assuming 'some_belief_array' and 'prediction' are your data sources
for belief, prediction in msp_belief_token_prediction.items():
    key = create_string_key(belief)
    msp_belief_token_prediction_str[key] = prediction


n_batch, n_ctx = transformer_inputs.shape
for b in range(n_batch):
    for c in range(n_ctx):
        input = transformer_inputs[b, :c]
        belief = transformer_input_beliefs[b, c, :]
        belief_set = string_key(belief.cpu().numpy())
        
        next_token_prediction = msp_belief_token_prediction_str.get(belief_set)
        if next_token_prediction is not None:
            print(f"Input: {input}, Belief: {belief}, Next Token Prediction: {next_token_prediction}")
        else:
            print("No prediction found for this belief set.")

import pandas as pd
# for every input we need
# 1) DONE activations  # [batch, n_ctx, d_model]
# 2) belief representation (this is the projection of the activations)
# 3) next token prediction
# 4) ground truth belief
print(acts.shape)
print(belief_predictions.shape)

print(transformer_input_beliefs.shape)
df = pd.DataFrame(columns=["Input", "Ground Truth Belief", "Ground Truth Next Token Prediction", "Transformer Belief"])
for b in range(n_batch):
    for c in range(n_ctx):
        input = transformer_inputs[b, :c]
        input_str = ''.join([str(int(x)) for x in input.cpu().numpy()])
        activation = acts[b, c, :]
        ground_truth_belief = transformer_input_beliefs[b, c, :].cpu().numpy()
        ground_truth_next_token_prediction = msp_belief_token_prediction_str.get(string_key(ground_truth_belief))
        transformer_belief = belief_predictions[b, c, :]

        # put this all into a pandas dataframe
        new_row = {"Input": input_str,
                   "Ground Truth Belief": ground_truth_belief,
                   "Ground Truth Next Token Prediction": ground_truth_next_token_prediction,
                   "Transformer Belief": transformer_belief,
                   "Activation": activation.cpu().numpy()}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# go row by row and compute the L2 distance between the ground truth belief and the transformer belief
for i in range(len(df)):
    df.loc[i, 'Error'] = np.mean((df.loc[i, 'Ground Truth Belief'] - df.loc[i, 'Transformer Belief']) ** 2)

#%%
import seaborn as sns
#histogram of errors
sns.histplot(df['Error'], kde=False, alpha=0.5, element='step', stat='density', fill=False)
plt.xlim(0, 1)
plt.show()


# %%
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# Assuming df is your existing DataFrame and is populated
n_rows = len(df)  # Ensure we have the correct number of rows

# Function to convert data for distance calculation
def prepare_data(column):
    return np.stack(df[column].apply(np.array))

# Prepare data arrays
ground_truth_beliefs = prepare_data('Ground Truth Belief')
next_token_predictions = prepare_data('Ground Truth Next Token Prediction')
transformer_beliefs = prepare_data('Transformer Belief')
activations = prepare_data('Activation')

# Compute pairwise distances using cdist
gt_belief_distances = cdist(ground_truth_beliefs, ground_truth_beliefs, metric='euclidean')
next_token_pred_distances = cdist(next_token_predictions, next_token_predictions, metric='euclidean')
transformer_belief_distances = cdist(transformer_beliefs, transformer_beliefs, metric='euclidean')
activation_distances = cdist(activations, activations, metric='euclidean')

# Generate Input Pairs correctly matching the shape of upper triangular matrix without diagonal
#input_pairs = [f"{df.iloc[i]['Input']} - {df.iloc[j]['Input']}" for i in range(n_rows) for j in range(i + 1, n_rows)]

# Create DataFrame from distances
distance_data = {
    #"Input Pair": input_pairs,
    "Ground Truth Belief Distance": gt_belief_distances[np.triu_indices(n_rows, k=1)],
    "Ground Truth Next Token Prediction Distance": next_token_pred_distances[np.triu_indices(n_rows, k=1)],
    "Transformer Belief Distance": transformer_belief_distances[np.triu_indices(n_rows, k=1)],
    "Activation Distance": activation_distances[np.triu_indices(n_rows, k=1)]
}

distance_df = pd.DataFrame(distance_data)

# Print shapes to verify correctness
for i, v in distance_data.items():
    print(i, v.shape)  # All should match now


#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]
sns.set_context("talk")

dims = 4
largest_distance = np.max(distance_df['Ground Truth Belief Distance'])

# Setting up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)  # Share y-axis across subplots

# Define the pairs of columns to plot
columns_pairs = [
    ('Ground Truth Belief Distance', 'Transformer Belief Distance'),
    ('Ground Truth Next Token Prediction Distance', 'Transformer Belief Distance'),
    #('Ground Truth Belief Distance', 'Activation Distance'),
    #('Ground Truth Next Token Prediction Distance', 'Activation Distance')
]

# Iterate over each subplot and pair of columns
for ax, (x_col, y_col) in zip(axs.flat, columns_pairs):
    x = distance_df[x_col]
    y = distance_df[y_col]
    
    # Plot 2D histogram with density
    h = ax.hist2d(x, y, bins=25, cmap='binary', density=True)
    #plt.colorbar(h[3], ax=ax)

    # Fit a line to the data
    slope, intercept = np.polyfit(x, y, 1)
    best_fit_line = slope * x + intercept

    # Calculate R-squared
    determination = np.corrcoef(x, y)[0, 1]**2

    # Plot the best fit line
    ax.plot(x, best_fit_line, color='red', linestyle='dotted', alpha=0.5, label=f'Best Fit Line\n$R^2 = {determination:.2f}$')
    ax.legend()

    # Set titles and labels
    #ax.set_title(f'{x_col} vs. {y_col}')
    ax.set_xlabel(x_col)
    if ax is axs[0]:
        ax.set_ylabel(y_col)



    #ax.set_xlim(0, largest_distance)
    #ax.set_ylim(0, largest_distance)

# Adjust layout
plt.tight_layout()
plt.show()


#%%

# %%
_, activations = model.run_with_cache(transformer_inputs, names_filter=lambda x: 'resid' in x or 'ln_final.hook_normalized' in x)

relevant_activation_keys = ['blocks.0.hook_resid_pre', 'blocks.0.hook_resid_post', 'blocks.1.hook_resid_post', 'blocks.2.hook_resid_post', 'blocks.3.hook_resid_post', 'ln_final.hook_normalized']
keys_legend_names = ['0', '1', '2', '3', '4', 'ln_final']
key_legend_dict = {k: keys_legend_names[i] for i, k in enumerate(relevant_activation_keys)}
#only keep the relevant activations
activations = {k: activations[k] for k in relevant_activation_keys}
num_acts = len(activations)
print(num_acts)
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
set_inds = set(range(num_acts))

for s in powerset(set_inds):
    print(s)

activations
# %%
from tqdm import tqdm
# import pallete
palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]

results_df = pd.DataFrame(columns=["Input", "Ground Truth Belief", "Ground Truth Next Token Prediction", "Transformer Belief", 'Error', 'activation_type'])
for acts_ in tqdm(activations):
    acts = activations[acts_]
    regression, belief_predictions = run_activation_to_beliefs_regression(acts, transformer_input_beliefs)


    df = pd.DataFrame(columns=["Input", "Ground Truth Belief", "Ground Truth Next Token Prediction", "Transformer Belief", "activation_type"])
    for b in range(n_batch):
        for c in range(n_ctx):
            input = transformer_inputs[b, :c]
            input_str = ''.join([str(int(x)) for x in input.cpu().numpy()])
            activation = acts[b, c, :]
            ground_truth_belief = transformer_input_beliefs[b, c, :].cpu().numpy()
            ground_truth_next_token_prediction = msp_belief_token_prediction_str.get(string_key(ground_truth_belief))
            transformer_belief = belief_predictions[b, c, :]



            # put this all into a pandas dataframe
            new_row = {"Input": input_str,
                    "Ground Truth Belief": ground_truth_belief,
                    "Ground Truth Next Token Prediction": ground_truth_next_token_prediction,
                    "Transformer Belief": transformer_belief,
                    "Activation": activation.cpu().numpy()}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # go row by row and compute the L2 distance between the ground truth belief and the transformer belief
    for i in range(len(df)):
        df.loc[i, 'Error'] = np.mean((df.loc[i, 'Ground Truth Belief'] - df.loc[i, 'Transformer Belief']) ** 2)
    df['activation_type'] = key_legend_dict[acts_]

    # add the results to the results_df
    results_df = pd.concat([results_df, df], ignore_index=True)

#make acts concat
acts = torch.concatenate(list(activations.values()), dim=-1)
regression, belief_predictions = run_activation_to_beliefs_regression(acts, transformer_input_beliefs)


df = pd.DataFrame(columns=["Input", "Ground Truth Belief", "Ground Truth Next Token Prediction", "Transformer Belief"])
for b in range(n_batch):
    for c in range(n_ctx):
        input = transformer_inputs[b, :c]
        input_str = ''.join([str(int(x)) for x in input.cpu().numpy()])
        activation = acts[b, c, :]
        ground_truth_belief = transformer_input_beliefs[b, c, :].cpu().numpy()
        ground_truth_next_token_prediction = msp_belief_token_prediction_str.get(string_key(ground_truth_belief))
        transformer_belief = belief_predictions[b, c, :]



        # put this all into a pandas dataframe
        new_row = {"Input": input_str,
                "Ground Truth Belief": ground_truth_belief,
                "Ground Truth Next Token Prediction": ground_truth_next_token_prediction,
                "Transformer Belief": transformer_belief,
                "Activation": activation.cpu().numpy()}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# go row by row and compute the L2 distance between the ground truth belief and the transformer belief
for i in range(len(df)):
    df.loc[i, 'Error'] = np.mean((df.loc[i, 'Ground Truth Belief'] - df.loc[i, 'Transformer Belief']) ** 2)

df['activation_type'] = 'concat'

# add the results to the results_df
results_df = pd.concat([results_df, df], ignore_index=True)
# sns histogram of errors colored by type
g = sns.FacetGrid(results_df, col='activation_type', col_wrap=3)
g.map(sns.histplot, 'Error', kde=False, alpha=1., stat='density', fill=True, bins=25)
#plt.xlim(0, 1)
plt.show()

# make a box plot of mean error for each activation type
palette = sns.color_palette("viridis", n_colors=len(results_df['activation_type'].unique())-1)[::-1] + ["black"]
sns.boxplot(x='activation_type', y='Error', data=results_df, palette=palette)
plt.show()

# make a lineplot with errors
sns.lineplot(x='activation_type', y='Error', data=results_df, palette=palette)
plt.show()

# %%
# use sns to make a plot of the mean error for each activation type with reversed gradient colors
mean_errors = results_df.groupby('activation_type', as_index=True).Error.mean()
print(mean_errors)
palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]
sns.barplot(x='activation_type', y='Error', data=mean_errors, palette=palette)
plt.show()


#%%
sns.violinplot(x='activation_type', y='Error', data=results_df, palette=palette, split=True, inner=False, fill=False)
plt.show()

# %%

sns.boxenplot(x='activation_type', y='Error', data=results_df, palette=palette, 
              showfliers=False)
plt.show()

#%%
sns.boxplot(x='activation_type', y='Error', data=results_df, palette=palette, 
            showfliers=True)
plt.show()

# %%
#sns histogram
sns.histplot(x='Error', data=results_df, kde=False, alpha=1, element='step', stat='density', fill=False,
             hue='activation_type', palette=palette, bins=25)
plt.show()

sns.kdeplot(x='Error', data=results_df, hue='activation_type', palette=palette, 
            fill=True, alpha=0.5, linewidth=0)
plt.xlim(0, 1.25)
plt.show()

# %%
# Setup the FacetGrid
import seaborn as sns

# Setup the FacetGrid with vertical histograms
g = sns.FacetGrid(results_df, row='activation_type', height=4, aspect=2, sharex=False)
g.map(sns.histplot, 'Error', element='step', stat='density', bins=20, color='blue', fill=True, vertical=True)

g.set(xticks=[])
g.fig.subplots_adjust(wspace=0.2)  # Adjust space between plots
# %%
import seaborn as sns

# Setup the FacetGrid
g = sns.FacetGrid(results_df, row='activation_type', height=4, aspect=2, sharex=False)
# Mapping a vertical KDE plot
g.map(sns.kdeplot, 'Error', vertical=True, fill=True)

# %%
# make plot wide
palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]
g = sns.displot(data=results_df, y='Error', col='activation_type', kind='hist', 
                fill=True, aspect=.3, log_scale=True, bins=20, palette=palette, hue='activation_type',
                alpha=.7, linewidth=0)
#ylim
g.set(ylim=(0, 1.5))

g.set(xticks=[])
#g.set(yticks=[])  # Remove y-axis ticks
g.fig.subplots_adjust(wspace=-0.)  # Make plots closer to each other

# %%

palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]
g = sns.displot(data=results_df, x='Error', row='activation_type', kind='hist', 
                fill=True, log_scale=True, bins=20, palette=palette, hue='activation_type',
                linewidth=0, aspect=1/.3, facet_kws={'sharex': True}, legend=False, height=.7)
# remove titles
g.set_titles("")
g.set(yticks=[])  # Remove y-axis ticks
g.set(xticks=[0, .01,0.1, 1.0])
g.set(xlim=(0, .5))
# get rid of x lines
g.fig.subplots_adjust(wspace=-1.6)  # Make plots closer to each other
# %%
palette = sns.color_palette("viridis", n_colors=len(mean_errors)-1)[::-1] + ["black"]
g = sns.displot(data=results_df, x='Error', row='activation_type', kind='hist', 
                fill=True, log_scale=False, bins=20, palette=palette, hue='activation_type',
                linewidth=0, aspect=1/.3, facet_kws={'sharex': True}, legend=False, height=.7)
# remove titles
g.set_titles("")
g.set(yticks=[])  # Remove y-axis ticks
g.set(xticks=[0, .01,0.1, 1.0])
g.set(xlim=(0, 1.0))
# get rid of x lines
g.fig.subplots_adjust(wspace=-1.6)  # Make plots closer to each other
# %%
# plot mean error vs. activation type with error bars
order = ['0', '1', '2', '3', '4', 'ln_final', 'concat']
mean_errors = results_df.groupby('activation_type')['Error'].median().reindex(order)
error_bar = .25
lower_error = results_df.groupby('activation_type')['Error'].quantile(error_bar).reindex(order)
upper_error = results_df.groupby('activation_type')['Error'].quantile(1-error_bar).reindex(order)
error_bars = [mean_errors - lower_error, upper_error - mean_errors]

palette = sns.color_palette("viridis", n_colors=len(mean_errors))[::-1] + ["black"]
colors = palette[:len(mean_errors)]
labels = ['Embed', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'LN Final', 'Concat']
for a, activation_type in enumerate(mean_errors.index):
    plt.errorbar(x=a, y=mean_errors[activation_type], yerr=[[error_bars[0][activation_type]], [error_bars[1][activation_type]]], fmt='o', color=colors[a], ecolor=colors[a], capsize=5, markersize=5)
plt.xlabel('Activation Type')
plt.ylabel('Mean Squared Error')
plt.xticks(ticks=range(len(mean_errors)), labels=labels, rotation=45)
plt.show()
# %%
# plot mean error vs. activation type with error bars
order = ['0', '1', '2', '3', '4', 'ln_final', 'concat']
mean_errors = results_df.groupby('activation_type')['Error'].mean().reindex(order)
std_errors = results_df.groupby('activation_type')['Error'].sem().reindex(order)

palette = sns.color_palette("viridis", n_colors=len(mean_errors))[::-1] + ["black"]
colors = palette[:len(mean_errors)]
labels = ['Embed', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'LN Final', 'Concat']
for a, activation_type in enumerate(mean_errors.index):
    plt.errorbar(x=a, y=mean_errors[activation_type], yerr=std_errors[activation_type], fmt='o', color=colors[a], ecolor=colors[a], capsize=5, markersize=5)
plt.xlabel('Activation Type')
plt.ylabel('Mean Squared Error')
plt.xticks(ticks=range(len(mean_errors)), labels=labels, rotation=45)
#plt.ylim(0, 0.15)

# %%
# %%
# plot mean error vs. activation type with error bars
order = ['0', '1', '2', '3', '4', 'ln_final', 'concat']
mean_errors = results_df.groupby('activation_type')['Error'].mean().reindex(order)
std_errors = results_df.groupby('activation_type')['Error'].sem().reindex(order)

palette = sns.color_palette("viridis", n_colors=len(mean_errors))[::-1] + ["black"]
colors = palette[:len(mean_errors)]
labels = ['Embed', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'LN Final', 'Concat']
for a, activation_type in enumerate(mean_errors.index):
    plt.errorbar(x=a, y=mean_errors[activation_type], yerr=std_errors[activation_type], fmt='o', color='k', ecolor='k', capsize=5, markersize=5)
plt.xlabel('Activation Type')
plt.ylabel('Mean Squared Error')
plt.xticks(ticks=range(len(mean_errors)), labels=labels, rotation=45)
#plt.ylim(0, 0.15)
# %%
