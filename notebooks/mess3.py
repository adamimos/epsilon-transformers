#%%
%load_ext autoreload
%autoreload 2

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('../src/')
from simple_transformer import MultilayerTransformer, initialize_weights, train_model, train_hooked_model
from markov_utilities import (
    calculate_sequence_probabilities,
    compute_myopic_entropy_from_MSP,
    epsilon_machine_to_graph,
    get_recurrent_subgraph,
    to_mixed_state_presentation,
    get_recurrent_subgraph,
    calculate_empirical_sequence_probabilities,
    create_transition_matrix,
    to_probability_distributions
)


from entropy_analysis import (
    compute_conditional_entropy,
    compute_empirical_conditional_entropy,
    inverse_binary_entropy,

)

from error_analysis import (
    compute_minimum_error,
)



from visualization import visualize_graph
from processes import RRXORProcess, GoldenMeanProcess, ZeroOneRProcess, EvenProcess, Mess3Process

# %%
Z1R = Mess3Process()

# visualize the epsilon machine
graph = epsilon_machine_to_graph(Z1R.T, Z1R.state_names)
visualize_graph(graph, layout="spectral", draw_mixed_state=True, pdf="zero_one_r_epsilon.pdf")

# %%
MSP = to_mixed_state_presentation(Z1R.T, threshold=1e-100, max_depth=8)
#G_MSP = epsilon_machine_to_graph(MSP)
#visualize_graph(G_MSP, layout='spring', draw_edge_labels=True,
#                                    draw_mixed_state=True, draw_color=False, pdf="zero_one_r_msp.pdf")
# %%
from transformer_lens import HookedTransformer, HookedTransformerConfig

config = HookedTransformerConfig(
    d_model=64,
    d_head=8,
    n_layers=24,
    n_ctx=7,
    n_heads=24,
    d_mlp=64,
    d_vocab=3,
    act_fn='relu',
    use_attn_scale=True,
    #normalization_type=None,
    attention_dir='causal',
    attn_only=False,
    seed=42,
    init_weights=True,
    device='cpu'
)

model = HookedTransformer(config)
# Define a config for the transformer and training
train_config = {
    # training config
    'batch_size': 64,
    'sequence_length': 5000,
    'num_epochs': 200,
    'learning_rate': 1.5e-2,
    'weight_decay': 0,
    'patience': 500,
    'factor': 0.5
}

# Generate sequence data with positions
train_loader, test_loader, sequence_positions = Z1R.prepare_data(train_config['sequence_length'], model.cfg.n_ctx,
                                                                   split_ratio=0.8, batch_size=train_config['batch_size'],
                                                                   with_positions=True)

print(f"The number of batches in the training set is {len(train_loader)}")


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=train_config['patience'], factor=train_config['factor'], verbose=True)

model = train_hooked_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=train_config['num_epochs'], verbose=True)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# make the seaborn context prettier
sns.set_context('talk')

# define loss function as cross entropy loss, but no reduction
crossEntropy = nn.CrossEntropyLoss(reduction='none')

model.eval()
with torch.no_grad():
    # For train data
    train_loss = []
    train_accuracy = []
    for data, target in train_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = crossEntropy(output.view(-1, 3), target.view(-1))
        # Reshape loss back to original shape before saving
        train_loss.append(loss.view(data.shape[0], data.shape[1]).cpu().numpy())
        # prediction is argmax over last dimension, which is 2
        pred = output.data.max(2, keepdim=True)[1]
        correct = pred.squeeze() ==  data
        train_accuracy.append(correct.cpu().numpy())

    # For test data
    test_loss = []
    test_accuracy = []
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        output = model(data)
        loss = crossEntropy(output.view(-1, 3), target.view(-1))
        test_loss.append(loss.view(data.shape[0], data.shape[1]).cpu().numpy())
        pred = output.data.max(2, keepdim=True)[1]
        correct = pred.squeeze() ==  data
        test_accuracy.append(correct.cpu().numpy())

train_loss = np.concatenate(train_loss)
train_accuracy = np.concatenate(train_accuracy)
test_loss = np.concatenate(test_loss)
test_accuracy = np.concatenate(test_accuracy)

print('Train Loss: ', np.mean(train_loss))
print('Train Accuracy: ', np.mean(train_accuracy))
print('Test Loss: ', np.mean(test_loss))
print('Test Accuracy: ', np.mean(test_accuracy))

# Reshape the data to long format
train_loss_df = pd.DataFrame(train_loss).melt(var_name='context_position', value_name='loss')
train_loss_df['type'] = 'train'
test_loss_df = pd.DataFrame(test_loss).melt(var_name='context_position', value_name='loss')
test_loss_df['type'] = 'test'

# add 1 to context pos
train_loss_df['context_position'] += 1
test_loss_df['context_position'] += 1

# convert to bits from nats
train_loss_df['loss'] = np.log2(np.exp(train_loss_df['loss']))
test_loss_df['loss'] = np.log2(np.exp(test_loss_df['loss']))

# Concatenate the dataframes
loss_df = pd.concat([train_loss_df, test_loss_df])
# %%
# Compute myopic entropy
myopic_entropy = compute_myopic_entropy_from_MSP(MSP,8)

# Create a DataFrame for myopic entropy
myopic_entropy_df = pd.DataFrame({
    'context_position': np.arange(len(myopic_entropy)),
    'loss': myopic_entropy,
    'type': 'myopic entropy'
})

# Concatenate the dataframes
loss_df = pd.concat([train_loss_df, test_loss_df, myopic_entropy_df])

# Define colors for each line
colors = {'train': 'blue', 'test': 'green', 'myopic entropy': 'red'}

# Plot using seaborn
plt.figure(figsize=(10, 6))
plt.grid(True)  # Add a grid for better readability
# make grid lighter
plt.gca().set_axisbelow(True)
plt.grid(which='major', color='#999999', linestyle='-', alpha=0.1)
sns.lineplot(data=loss_df, x='context_position', y='loss', hue='type', palette=colors)
plt.ylabel('Loss [bits]')
plt.xlabel('Context Window Position')
plt.title('Cross Entropy vs Position for 01R Process')
plt.show()
# %%
probs = to_probability_distributions(Z1R.T,8,1e-15)

#%%
import pandas as pd
import plotly.express as px
import ternary
df = pd.DataFrame([100*p for p in probs], columns=['A', 'B', 'C'])
fig, tax = ternary.figure(scale=100)

# Axis labels. (See below for corner labels.)
fontsize = 14
offset = 0.08
tax.left_axis_label("State A %", fontsize=fontsize, offset=offset)
tax.right_axis_label("State B %", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("State C %", fontsize=fontsize, offset=-offset)


tax.scatter(df[['A', 'B', 'C']].values, s=1,c='k')
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')
tax.show()
# %%
# get some training data and run it through the model
from tqdm import tqdm
datas = []
resids = []
for data, target in tqdm(train_loader):
    _, cache = model.run_with_cache(data, names_filter=lambda x: 'resid_post' in x)
    datas.extend(data)
    resids.append(cache["resid_post", model.cfg.n_layers-1])
resids = torch.concat(resids)
print(resids.shape) #(batch, ctx_pos, res_dim)

MSP_states = []
for d in datas:
    this_data = []
    from_state = 0
    for p in d:
        emission = p
        transition_probs = MSP[p,from_state,:]
        # find all nonzero index
        nonzero_indices = np.nonzero(transition_probs)[0]
        #print(nonzero_indices)
        if len(nonzero_indices) != 1:
            print('warning',nonzero_indices)
        from_state = nonzero_indices[0]
        this_data.append(from_state)
    MSP_states.append(this_data)

# %%
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from collections import Counter


resids_reshaped = resids.view(resids.shape[0]*resids.shape[1], -1)
MSP_states= torch.tensor(MSP_states)
MSP_states_reshaped = MSP_states.view(-1)

state_prob_dict = dict(zip(range(len(probs)),probs))
state_counts = Counter([int(x) for x in MSP_states_reshaped])
print(state_counts)

print(resids_reshaped.shape, MSP_states_reshaped.shape)

# for each of the MSP_states_reshaped, map it to  probs using state_prob_dict
# and weight it according to its frequency of occurrence
target_distributions = np.array([state_prob_dict[int(x)] for x in MSP_states_reshaped])
weights = np.array([state_counts[int(x)] for x in MSP_states_reshaped])
print(target_distributions.shape)


# Initialize MultiOutputRegressor with a LinearRegression model
model = MultiOutputRegressor(LinearRegression())

# Fit the model to your data using the weights
model.fit(resids_reshaped, target_distributions)

# Predict the distributions for new data
predicted_distributions = model.predict(resids_reshaped)

 # %%
import plotly.express as px
import pandas as pd

# Convert the predicted_distributions and MSP_states_reshaped to a DataFrame
df = pd.DataFrame(np.concatenate((predicted_distributions, MSP_states_reshaped.reshape(-1,1)), axis=1), columns=['x', 'y', 'z', 'label'])
df['label'] = df['label'].astype(str)

# Create a color sequence based on the x, y, z values as RGB
df['color'] = df.apply(lambda row: 'rgb({},{},{})'.format(row['x']*255, row['y']*255, row['z']*255), axis=1)

# Create a 3D scatter plot using plotly express with color based on the RGB values
fig = px.scatter_3d(df, x='x', y='y', z='z', color='color')
fig.show()

# Project the data to the x+y+z=1 plane
df['x_proj'] = df['x'] / (df['x'] + df['y'] + df['z'])
df['y_proj'] = df['y'] / (df['x'] + df['y'] + df['z'])

# Plot the projected data on the 2D plane
# Use the same RGB colorscale
fig = px.scatter(df, x='x_proj', y='y_proj', color='color')
fig.show()

# %%
import plotly.graph_objects as go
import pandas as pd

# Convert the predicted_distributions and MSP_states_reshaped to a DataFrame
df = pd.DataFrame(np.concatenate((predicted_distributions, MSP_states_reshaped.reshape(-1,1)), axis=1), columns=['x', 'y', 'z', 'label'])
df['label'] = df['label'].astype(str)

# Create a color sequence based on the x, y, z values as RGB
df['color'] = df.apply(lambda row: 'rgb({},{},{})'.format(int(row['x']*255), int(row['y']*255), int(row['z']*255)), axis=1)

# Create a 3D scatter plot using plotly graph_objects with color based on the RGB values
fig = go.Figure(data=[go.Scatter3d(x=df['x'], y=df['y'], z=df['z'], mode='markers', marker=dict(color=df['color']))])
fig.show()

# Project the data to the x+y+z=1 plane
df['x_proj'] = df['x'] / (df['x'] + df['y'] + df['z'])
df['y_proj'] = df['y'] / (df['x'] + df['y'] + df['z'])

# Plot the projected data on the 2D plane
# Use the same RGB colorscale
fig = go.Figure(data=[go.Scatter(x=df['x_proj'], y=df['y_proj'], mode='markers', marker=dict(color=df['color']))])
fig.show()
# %%
probs_to_plot = np.array([np.array(x) for x in probs])
df_probs = pd.DataFrame(probs_to_plot, columns=['x', 'y', 'z'])
df_probs['label'] = [str(x) for x in range(len(probs_to_plot))]

fig = px.scatter_3d(df_probs, x='x', y='y', z='z', color='label', color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()

df_probs['x_proj'] = df_probs['x'] / (df_probs['x'] + df_probs['y'] + df_probs['z'])
df_probs['y_proj'] = df_probs['y'] / (df_probs['x'] + df_probs['y'] + df_probs['z'])

fig = px.scatter(df_probs, x='x_proj', y='y_proj', color='label', color_discrete_sequence=px.colors.qualitative.Plotly)
fig.show()

# %%
import ternary

# Create a ternary plot
figure, tax = ternary.figure(scale=1.0)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=0.2, color="blue")

# Plot the data with color based on the x, y, z values acting as RGB
colors = df_probs.apply(lambda row: [row['x'], row['y'], row['z']], axis=1)
tax.scatter(df_probs[['x', 'y', 'z']].values, marker='s', c=colors.tolist(), label="Projected Probabilities")

# Set labels and title
tax.set_title("Ternary Plot of Projected Probabilities", fontsize=20)
tax.left_axis_label("x", fontsize=16)
tax.right_axis_label("y", fontsize=16)
tax.bottom_axis_label("z", fontsize=16)

# Show the plot
tax.show()

# %%
import pandas as pd
import plotly.express as px
import ternary
df = pd.DataFrame([100*p for p in probs], columns=['0', '1', 'R'])
df['label'] = [str(x) for x in range(len(probs))]
fig, tax = ternary.figure(scale=100)

# Axis labels. (See below for corner labels.)
fontsize = 14
offset = 0.08
tax.left_axis_label("State R %", fontsize=fontsize, offset=offset)
tax.right_axis_label("State 0 %", fontsize=fontsize, offset=offset)
tax.bottom_axis_label("State 1 %", fontsize=fontsize, offset=-offset)

# Add colors and legend
colors = df_probs.apply(lambda row: [row['x'], row['y'], row['z']], axis=1)

tax.scatter(df[['0', '1', 'R']].values, c=colors, label="Projected Probabilities",s=1)


tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')
tax.show()
# %%
import ternary

# Convert the predicted_distributions and MSP_states_reshaped to a DataFrame
df = pd.DataFrame(np.concatenate((predicted_distributions, MSP_states_reshaped.reshape(-1,1)), axis=1), columns=['x', 'y', 'z', 'label'])
df[['x', 'y', 'z']] = df[['x', 'y', 'z']] * 100

df['label'] = df['label'].astype(int).astype(str)


# Create a color sequence based on the unique labels in ascending order
color_sequence = px.colors.qualitative.Plotly[:len(df['label'].unique())]

# Create a ternary plot using the ternary library
cpred = np.clip(predicted_distributions, 0, 1)
colors = [list(p) for p in cpred]
fig, tax = ternary.figure(scale=100)
tax.scatter(df[['x', 'y', 'z']].values, color=colors, label="Projected Probabilities",s=1)
tax.boundary(linewidth=1)
tax.gridlines(multiple=10, color="gray")
tax.ticks(axis='lbr', linewidth=1, multiple=20)
tax.get_axes().axis('off')
tax.show()

# %%
import matplotlib.pyplot as plt

# Create a new figure
fig = plt.figure(figsize=(15, 7))

# Add first subplot for the first ternary plot
ax1 = fig.add_subplot(121)

# Create a ternary plot in the first subplot
tax1 = ternary.TernaryAxesSubplot(ax=ax1, scale=100)
df2 = pd.DataFrame([100*p for p in probs], columns=['0', '1', 'R'])
df2['label'] = [str(x) for x in range(len(probs))]
tax1.scatter(df2[['0', '1', 'R']].values, color=
             df_probs.apply(lambda row: [row['x'], row['y'], row['z']], axis=1),
             label="Projected Probabilities",s=2)
tax1.boundary(linewidth=1)
tax1.gridlines(multiple=10, color="gray")
tax1.ticks(axis='lbr', linewidth=1, multiple=20)
tax1.get_axes().axis('off')
tax1.set_title("Ground Truth")  # Add title to the first plot

# Add second subplot for the second ternary plot
ax2 = fig.add_subplot(122)

# Create a ternary plot in the second subplot
tax2 = ternary.TernaryAxesSubplot(ax=ax2, scale=100)
colors = [list(state_prob_dict[int(x)]) for x in MSP_states_reshaped]
tax2.scatter(df[['x', 'y', 'z']].values, color=colors,s=2)
tax2.gridlines(multiple=10, color="gray")
tax2.ticks(axis='lbr', linewidth=1, multiple=20)
tax2.get_axes().axis('off')
tax2.set_title("Residual Stream")  # Add title to the second plot

# Show the figure with the two ternary plots side by side
plt.show()



# %%
