from typing import Iterator, List, Tuple, Dict
from sklearn.linear_model import LinearRegression
from transformer_lens import HookedTransformer
from jaxtyping import Float
import numpy as np
import torch
from sklearn.decomposition import PCA
from epsilon_transformers.process.GHMM import GHMM, markov_approximation
from epsilon_transformers.process.transition_matrices import get_matrix_from_args
from epsilon_transformers.process.GHMM import TransitionMatrixGHMM
from epsilon_transformers.training.networks import RNNWrapper
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
import json
from epsilon_transformers.analysis.load_data import S3ModelLoader
import io



def get_sweep_type(run_id):
    """Determine sweep type from run_id."""
    if 'tom' in run_id:
        return 'tom'
    elif 'post' in run_id or 'mess3' in run_id:
        return 'post'
    return 'fanizza'

def get_beliefs_for_nn_inputs(
    nn_inputs: Float[torch.Tensor, "batch n_ctx"], 
    msp_belief_index: Dict[Tuple[float, ...], int], 
    tree_paths: List[List[int]], 
    tree_beliefs: List[List[float]],
    tree_unnormalized_beliefs: List[List[float]] | None,
    probs_dict: Dict[Tuple[int, ...], float | None]
) -> Tuple[Float[torch.Tensor, "batch n_ctx belief_dim"], Float[torch.Tensor, "batch n_ctx"]]:
    """
    Converts neural network input sequences into their corresponding belief states and indices.
    
    Args:
        nn_inputs: Input sequences of shape (batch, n_ctx)
        msp_belief_index: Dictionary mapping belief state tuples to integer indices
        tree_paths: List of paths through the mixed state presentation tree, where each path is a list of integers
        tree_beliefs: List of belief states corresponding to each path, where each belief state is a list of floats
        tree_unnormalized_beliefs: List of unnormalized belief states corresponding to each path, where each unnormalized belief state is a list of floats
        probs_dict: Dictionary mapping input sequences to their corresponding probabilities
    Returns:
        Tuple containing:
        - X_beliefs: Tensor of belief states for each position in each sequence, shape (batch, n_ctx, belief_dim)
        - X_belief_indices: Tensor of integer indices for each belief state, shape (batch, n_ctx)
        - X_probs: Tensor of probabilities for each belief state, shape (batch, n_ctx) if probs_dict is not None, otherwise None
    """
    batch, n_ctx = nn_inputs.shape
    belief_dim = len(list(msp_belief_index.keys())[0])
    path_belief_dict = {tuple(path): belief for path, belief in zip(tree_paths, tree_beliefs)}
    X_beliefs = torch.zeros(batch, n_ctx, belief_dim, device=nn_inputs.device)
    X_belief_indices = torch.zeros(batch, n_ctx, dtype=torch.int, device=nn_inputs.device)


    if tree_unnormalized_beliefs is not None:
        X_unnormalized_beliefs = torch.zeros(batch, n_ctx, belief_dim, device=nn_inputs.device)
        path_unnormalized_belief_dict = {tuple(path): belief for path, belief in zip(tree_paths, tree_unnormalized_beliefs)}
    else:
        path_unnormalized_belief_dict = None

    if probs_dict is not None:
        X_probs = torch.zeros(batch, n_ctx, dtype=torch.float32, device=nn_inputs.device)
    
    for i in range(batch):
        for j in range(n_ctx):
            input_substring = tuple(nn_inputs[i, :j+1].cpu().numpy())
            full_string = tuple(nn_inputs[i].cpu().numpy())
            belief_state = path_belief_dict[input_substring].squeeze()
            belief_state = np.round(belief_state, 5)
            X_beliefs[i, j] = torch.tensor(belief_state, dtype=torch.float32, device=nn_inputs.device)
            X_belief_indices[i, j] = msp_belief_index[tuple(belief_state)]
            if probs_dict is not None:
                X_probs[i, j] = probs_dict[full_string]
            if path_unnormalized_belief_dict is not None:
                X_unnormalized_beliefs[i, j] = torch.tensor(path_unnormalized_belief_dict[input_substring].squeeze(), dtype=torch.float32, device=nn_inputs.device)

    if probs_dict is None and path_unnormalized_belief_dict is None:
        return X_beliefs, X_belief_indices
    elif probs_dict is None:
        return X_beliefs, X_belief_indices, X_unnormalized_beliefs
    elif path_unnormalized_belief_dict is None:
        return X_beliefs, X_belief_indices, X_probs
    else:
        return X_beliefs, X_belief_indices, X_probs, X_unnormalized_beliefs

def prepare_data_from_msp(msp, path_length):
    tree_paths = msp.paths
    tree_beliefs = msp.belief_states
    tree_unnormalized_beliefs = msp.unnorm_belief_states
    path_probs = msp.path_probs
    msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in tree_beliefs]
    msp_belief_index = {tuple(b): i for i, b in enumerate(set(msp_beliefs))}
    
    nn_paths = [x for x in tree_paths if len(x) == path_length]
    nn_inputs = torch.tensor(nn_paths, dtype=torch.int).clone().detach().to("cpu")

    probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}
    
    nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = get_beliefs_for_nn_inputs(
        nn_inputs,
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized_beliefs,
        probs_dict
    )
    
    return nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs

def prepare_msp_data(config, model_config):
    """Prepare MSP belief states and transformer inputs."""
    msp = get_msp(config)
    tree_paths = msp.paths
    tree_beliefs = msp.belief_states
    tree_unnormalized_beliefs = msp.unnorm_belief_states
    path_probs = msp.path_probs
    msp_beliefs = [tuple(round(b, 5) for b in belief.squeeze()) for belief in tree_beliefs]
    msp_belief_index = {tuple(b): i for i, b in enumerate(set(msp_beliefs))}
    
    nn_paths = [x for x in tree_paths if len(x) == model_config['n_ctx']]
    nn_inputs = torch.tensor(nn_paths, dtype=torch.int).clone().detach().to("cpu")

    probs_dict = {tuple(path): prob for path, prob in zip(tree_paths, path_probs)}
    
    nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = get_beliefs_for_nn_inputs(
        nn_inputs,
        msp_belief_index,
        tree_paths,
        tree_beliefs,
        tree_unnormalized_beliefs,
        probs_dict
    )
    
    return nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs

def run_activation_to_beliefs_regression(activations, ground_truth_beliefs, sample_weights=None):

    # make sure the first two dimensions are the same
    assert activations.shape[0] == ground_truth_beliefs.shape[0]
    assert activations.shape[1] == ground_truth_beliefs.shape[1]

    # detach whatever needs to be detached
    activations = activations.detach()
    ground_truth_beliefs = ground_truth_beliefs.detach()

    # flatten the activations
    batch_size, n_ctx, d_model = activations.shape
    #print(batch_size, n_ctx, d_model)
    belief_dim = ground_truth_beliefs.shape[-1]
    activations_flattened = activations.detach().reshape(-1, d_model) # [batch * n_ctx, d_model]
    ground_truth_beliefs_flattened = ground_truth_beliefs.view(-1, belief_dim) # [batch * n_ctx, belief_dim]
    sample_weights_flattened = sample_weights.view(-1) if sample_weights is not None else None

    # run the regression
    regression = LinearRegression()
    regression.fit(activations_flattened, ground_truth_beliefs_flattened, sample_weight=sample_weights_flattened)

    # get the belief predictions
    belief_predictions = regression.predict(activations_flattened) # [batch * n_ctx, belief_dim]
    belief_predictions = belief_predictions.reshape(batch_size, n_ctx, belief_dim)
    if sample_weights_flattened is not None:
        mse = (((torch.tensor(belief_predictions).view(-1, belief_dim) - ground_truth_beliefs_flattened)**2) * sample_weights_flattened.unsqueeze(-1)).mean()
    else:
        mse = ((torch.tensor(belief_predictions).view(-1, belief_dim) - ground_truth_beliefs_flattened)**2).mean()

    # cross-validation
    n_samples = activations_flattened.shape[0]
    n_train = n_samples // 2
    
    # Make sure we can reshape properly later
    n_train = (n_train // n_ctx) * n_ctx  # Make divisible by n_ctx
    
    sample_inds = torch.randperm(n_samples)
    train_inds = sample_inds[:n_train]
    test_inds = sample_inds[n_train:]
    sample_weights_cv = sample_weights_flattened[train_inds] if sample_weights_flattened is not None else None

    # fit the regression on the data_fit
    regression_cv = LinearRegression()
    regression_cv.fit(activations_flattened[train_inds], ground_truth_beliefs_flattened[train_inds], sample_weight=sample_weights_cv)

    # predict on the other half
    belief_predictions_cv = regression_cv.predict(activations_flattened[test_inds])
    n_test_batches = len(test_inds) // n_ctx
    belief_predictions_cv = belief_predictions_cv.reshape(n_test_batches, n_ctx, belief_dim)
    if sample_weights_flattened is not None:
        mse_cv = (((torch.tensor(belief_predictions_cv).view(-1, belief_dim) - ground_truth_beliefs_flattened[test_inds])**2) * sample_weights_flattened[test_inds].unsqueeze(-1)).mean()
    else:
        mse_cv = ((torch.tensor(belief_predictions_cv).view(-1, belief_dim) - ground_truth_beliefs_flattened[test_inds])**2).mean()
    

    shuffle_indices = torch.randperm(activations_flattened.shape[0])
    activations_flattened_shuffled = activations_flattened
    ground_truth_beliefs_flattened_shuffled = ground_truth_beliefs_flattened[shuffle_indices]
    shuffle_regression = LinearRegression()
    shuffle_regression.fit(activations_flattened_shuffled, ground_truth_beliefs_flattened_shuffled, sample_weight=sample_weights_flattened  )
    belief_predictions_shuffled = shuffle_regression.predict(activations_flattened_shuffled) # [batch * n_ctx, belief_dim]
    belief_predictions_shuffled = belief_predictions_shuffled.reshape(batch_size, n_ctx, belief_dim)
    if sample_weights_flattened is not None:
        mse_shuffled = (((torch.tensor(belief_predictions_shuffled).view(-1, belief_dim) - ground_truth_beliefs_flattened_shuffled)**2) * sample_weights_flattened.unsqueeze(-1)).mean()
    else:
        mse_shuffled = ((torch.tensor(belief_predictions_shuffled).view(-1, belief_dim) - ground_truth_beliefs_flattened_shuffled)**2).mean()

    

    return regression, belief_predictions, mse, mse_shuffled, belief_predictions_shuffled, mse_cv, belief_predictions_cv, test_inds


def run_model_with_cache(model, inputs, batch_size=500):
    inputs = inputs.clone().detach()
    all_hidden_states = []
    
    for i in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[i:i+batch_size]
        # Run RNN and get hidden states
        with torch.no_grad():  # Disable gradient tracking
            output, layer_states = model.forward_with_hidden(batch_inputs)
        all_hidden_states.append(layer_states)  # [num_layers, batch_size, seq_len, hidden_size]

    # Concatenate all hidden states along batch dimension
    all_hidden_states = torch.cat(all_hidden_states, dim=1)  # [num_layers, total_batch_size, seq_len, hidden_size]
    
    # Permute dimensions to match transformer format: [seq_len, batch_size, num_layers, hidden_size]
    all_hidden_states = all_hidden_states.permute(2, 1, 0, 3).detach()  # Add detach here
    
    return {'hidden_states': all_hidden_states}

def plot_belief_predictions(belief_predictions,
                            transformer_input_beliefs, 
                            transformer_input_belief_indices, 
                            ax,
                            type='tom',
                            title=None,
                            mode='both',
                            include_colorbar=False,
                            cv_test_inds = None):
    """
    Plots the belief predictions on the provided axes.

    Parameters:
    - belief_predictions (np.ndarray): Predicted belief states.
    - transformer_input_beliefs (torch.Tensor): True belief states from transformer inputs.
    - transformer_input_belief_indices (torch.Tensor): Indices of the belief states.
    - ax (matplotlib.axes.Axes): The axes on which to plot.
    - type (str): Type of plot ('tom', 'fanizza', or 'post').
    - title (str): Optional title for the subplot.
    - mode (str): What to plot ('true', 'predicted', 'both').
    - include_colorbar (bool): Whether to include a colorbar.
    
    Returns:
    - ax (matplotlib.axes.Axes): The axes with the plotted data.
    """
    belief_predictions_flat = belief_predictions.reshape(-1, belief_predictions.shape[-1])
    transformer_input_beliefs_flat = transformer_input_beliefs.reshape(-1, transformer_input_beliefs.shape[-1]).cpu().numpy()
    if cv_test_inds is not None:
        transformer_input_beliefs_flat = transformer_input_beliefs_flat[cv_test_inds]
    
    # Compute distances from origin for true beliefs
    distances = np.sqrt(np.sum(transformer_input_beliefs_flat[:, 1:]**2, axis=1))
    belief_indices = transformer_input_belief_indices.cpu().numpy()

    pca = PCA(n_components=2)
    transformer_input_beliefs_flat = pca.fit_transform(transformer_input_beliefs_flat)
    belief_predictions_flat = pca.transform(belief_predictions_flat)

    if cv_test_inds is not None:
        # flatten the indices
        belief_shape = belief_indices.shape
        belief_indices = belief_indices.flatten()
        belief_indices = belief_indices[cv_test_inds]
        # unflatten the indices
        belief_indices = belief_indices.reshape(-1, belief_shape[1])
    
    if type == 'tom':
        s = [1, .1]  # size of points
        alpha = [.1, .05]  # transparency of points
        com = False
        c = distances
    elif type == 'fanizza':
        s = [15, 1]
        alpha = [.5, .1]
        com = True
        c = belief_indices.flatten()
    elif type == 'post':
        s = [2, 2]
        alpha = [.1, .2]
        com = False
        c = distances

    if mode not in ['true', 'predicted', 'both']:
        raise ValueError("Mode must be 'true', 'predicted', or 'both'.")

    # Determine common limits if plotting both
    if mode == 'both':
        x_min = min(transformer_input_beliefs_flat[:, 0].min(), belief_predictions_flat[:, 0].min())
        x_max = max(transformer_input_beliefs_flat[:, 0].max(), belief_predictions_flat[:, 0].max())
        y_min = min(transformer_input_beliefs_flat[:, 1].min(), belief_predictions_flat[:, 1].min())
        y_max = max(transformer_input_beliefs_flat[:, 1].max(), belief_predictions_flat[:, 1].max())
    else:
        if mode == 'true':
            # add some padding
            pad_percent = 0.1
            x_range = transformer_input_beliefs_flat[:, 0].max() - transformer_input_beliefs_flat[:, 0].min()
            y_range = transformer_input_beliefs_flat[:, 1].max() - transformer_input_beliefs_flat[:, 1].min()
            x_min, x_max = transformer_input_beliefs_flat[:, 0].min() - pad_percent * x_range, transformer_input_beliefs_flat[:, 0].max() + pad_percent * x_range
            y_min, y_max = transformer_input_beliefs_flat[:, 1].min() - pad_percent * y_range, transformer_input_beliefs_flat[:, 1].max() + pad_percent * y_range
        else:  # mode == 'predicted'
            x_min, x_max = belief_predictions_flat[:, 0].min(), belief_predictions_flat[:, 0].max()
            y_min, y_max = belief_predictions_flat[:, 1].min(), belief_predictions_flat[:, 1].max()

    if mode in ['true', 'both']:
        # Plot true beliefs
        scatter0 = ax.scatter(transformer_input_beliefs_flat[:, 0], transformer_input_beliefs_flat[:, 1], 
                             c=c, alpha=alpha[0], s=s[0], cmap='viridis', label='True Beliefs')

    if mode in ['predicted', 'both']:
        # Plot predicted beliefs
        scatter1 = ax.scatter(belief_predictions_flat[:, 0], belief_predictions_flat[:, 1], 
                             c=c, alpha=alpha[1], s=s[1], cmap='viridis', label='Predicted Beliefs')
    
    if com and mode in ['both', 'predicted']:
        unique_indices = sorted(set(belief_indices.flatten()))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_indices)))
        for i, color in zip(unique_indices, colors):
            com_point = np.mean(belief_predictions_flat[belief_indices.flatten() == i], axis=0)
            ax.scatter(com_point[0], com_point[1], s=s[0], alpha=1, color=color, label=f'COM {i}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if title:
        ax.set_title(title)
    
    # Remove box and tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    
    # Add colorbar if requested and mode is both
    if include_colorbar and mode == 'both':
        norm = plt.Normalize(vmin=min(c), vmax=max(c))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Color Scale')
    
    # Optionally add legend
    if mode == 'both':
        ax.legend(loc='best')
    
    return ax


def plot_belief_predictions2(belief_predictions,
                            transformer_input_beliefs, 
                            transformer_input_belief_indices, 
                            ax,
                            type='tom',
                            title=None,
                            mode='both',
                            include_colorbar=False,
                            cv_test_inds = None):
    """
    Plots the belief predictions on the provided axes.

    Parameters:
    - belief_predictions (np.ndarray): Predicted belief states.
    - transformer_input_beliefs (torch.Tensor): True belief states from transformer inputs.
    - transformer_input_belief_indices (torch.Tensor): Indices of the belief states.
    - ax (matplotlib.axes.Axes): The axes on which to plot.
    - type (str): Type of plot ('tom', 'fanizza', or 'post').
    - title (str): Optional title for the subplot.
    - mode (str): What to plot ('true', 'predicted', 'both').
    - include_colorbar (bool): Whether to include a colorbar.
    
    Returns:
    - ax (matplotlib.axes.Axes): The axes with the plotted data.
    """
    belief_predictions_flat = belief_predictions.reshape(-1, belief_predictions.shape[-1])
    transformer_input_beliefs_flat = transformer_input_beliefs.reshape(-1, transformer_input_beliefs.shape[-1]).cpu().numpy()
    if cv_test_inds is not None:
        transformer_input_beliefs_flat = transformer_input_beliefs_flat[cv_test_inds]
    
    # Compute distances from origin for true beliefs
    distances = np.sqrt(np.sum(transformer_input_beliefs_flat[:, 1:]**2, axis=1))
    belief_indices = transformer_input_belief_indices.cpu().numpy()


    
    if cv_test_inds is not None:
        # flatten the indices
        belief_shape = belief_indices.shape
        belief_indices = belief_indices.flatten()
        belief_indices = belief_indices[cv_test_inds]
        # unflatten the indices
        belief_indices = belief_indices.reshape(-1, belief_shape[1])
    
    if type == 'tom':
        inds = [1,2]
        s = [1, .1]  # size of points
        alpha = [.1, .05]  # transparency of points
        com = False
        c = distances
    elif type == 'fanizza':
        inds = [2,3]
        s = [15, 1]
        alpha = [.5, .1]
        com = True
        c = belief_indices.flatten()
    elif type == 'post':
        inds = [1,2]
        s = [15, 5]
        alpha = [.1, .1]
        com = False
        c = distances

    if mode not in ['true', 'predicted', 'both']:
        raise ValueError("Mode must be 'true', 'predicted', or 'both'.")
    
    belief_dims = transformer_input_beliefs_flat.shape[-1]

    # if the inds are not in the range of the belief dimensions, then change the inds to the first and last available dimension
    if inds[0] not in range(belief_dims):
        inds[0] = 0
    if inds[1] not in range(belief_dims):
        inds[1] = 1

    # Determine common limits if plotting both
    if mode == 'both':
        x_min = min(transformer_input_beliefs_flat[:, inds[0]].min(), belief_predictions_flat[:, inds[0]].min())
        x_max = max(transformer_input_beliefs_flat[:, inds[0]].max(), belief_predictions_flat[:, inds[0]].max())
        y_min = min(transformer_input_beliefs_flat[:, inds[1]].min(), belief_predictions_flat[:, inds[1]].min())
        y_max = max(transformer_input_beliefs_flat[:, inds[1]].max(), belief_predictions_flat[:, inds[1]].max())
    else:
        if mode == 'true':
            # add some padding
            pad_percent = 0.1
            x_range = transformer_input_beliefs_flat[:, inds[0]].max() - transformer_input_beliefs_flat[:, inds[0]].min()
            y_range = transformer_input_beliefs_flat[:, inds[1]].max() - transformer_input_beliefs_flat[:, inds[1]].min()
            x_min, x_max = transformer_input_beliefs_flat[:, inds[0]].min() - pad_percent * x_range, transformer_input_beliefs_flat[:, inds[0]].max() + pad_percent * x_range
            y_min, y_max = transformer_input_beliefs_flat[:, inds[1]].min() - pad_percent * y_range, transformer_input_beliefs_flat[:, inds[1]].max() + pad_percent * y_range
        else:  # mode == 'predicted'
            x_min, x_max = belief_predictions_flat[:, inds[0]].min(), belief_predictions_flat[:, inds[0]].max()
            y_min, y_max = belief_predictions_flat[:, inds[1]].min(), belief_predictions_flat[:, inds[1]].max()

    if mode in ['true', 'both']:
        # Plot true beliefs
        scatter0 = ax.scatter(transformer_input_beliefs_flat[:, inds[0]], transformer_input_beliefs_flat[:, inds[1]], 
                             c=c, alpha=alpha[0], s=s[0], cmap='viridis', label='True Beliefs')

    if mode in ['predicted', 'both']:
        # Plot predicted beliefs
        scatter1 = ax.scatter(belief_predictions_flat[:, inds[0]], belief_predictions_flat[:, inds[1]], 
                             c=c, alpha=alpha[1], s=s[1], cmap='viridis', label='Predicted Beliefs')
    
    if com and mode in ['both', 'predicted']:
        unique_indices = sorted(set(belief_indices.flatten()))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_indices)))
        for i, color in zip(unique_indices, colors):
            com_point = np.mean(belief_predictions_flat[belief_indices.flatten() == i], axis=0)
            ax.scatter(com_point[inds[0]], com_point[inds[1]], s=s[0], alpha=1, color=color, label=f'COM {i}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    if title:
        ax.set_title(title)
    
    # Remove box and tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)
    
    # Add colorbar if requested and mode is both
    if include_colorbar and mode == 'both':
        norm = plt.Normalize(vmin=min(c), vmax=max(c))
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Color Scale')
    
    # Optionally add legend
    if mode == 'both':
        ax.legend(loc='best')
    
    return ax

def get_msp(run_config):
    T = get_matrix_from_args(**run_config['process_config'])
    ghmm = TransitionMatrixGHMM(T)
    ghmm.name = 'GHMM'
    msp = ghmm.derive_mixed_state_tree(depth=run_config['model_config']['n_ctx'])
    return msp

def markov_approx_msps(run_config, max_order=3):
    T = get_matrix_from_args(**run_config['process_config'])
    ghmm = TransitionMatrixGHMM(T)
    markov_data = []
    for order in tqdm(range(1, max_order+1), desc="Running Markov Approximations"):
        markov_approx = markov_approximation(ghmm, order)
        msp = markov_approx.derive_mixed_state_tree(depth=run_config['model_config']['n_ctx'])
        markov_data.append(prepare_data_from_msp(msp, run_config['model_config']['n_ctx']))
    return markov_data

def model_type(model) -> str:
    if isinstance(model, HookedTransformer):
        return "transformer"
    elif isinstance(model, RNNWrapper):
        return "rnn"
    else:
        raise ValueError(f"Model type {type(model)} not supported")



def generate_belief_state_and_activations(model: HookedTransformer, process: GHMM, num_sequences: int) -> Tuple[Float[np.ndarray, "num_samples n_ctx num_states"], Float[np.ndarray, "num_samples n_ctx d_model"]]:
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Create the Mixed State Presentation Tree
    msp_tree = process.derive_mixed_state_tree(depth=model.cfg.n_ctx + 1) # For really large models this should definately be parallelized
    
    belief_states = []
    activations = []
    for x in process.yield_emission_histories(sequence_len=model.cfg.n_ctx, num_sequences=num_sequences):
        belief_states.append(msp_tree.path_to_beliefs(x))
        _, cache = model.run_with_cache(torch.tensor([x], device=device))
        activations.append(cache['ln_final.hook_normalized'].squeeze(0).cpu())
    
    # Turn into tensors
    belief_states = np.stack(belief_states)
    activations = np.array(activations)

    return belief_states, activations

def find_msp_subspace_in_residual_stream(model: HookedTransformer, process: GHMM, num_sequences: int) -> Tuple[Float[np.ndarray, "num_tokens num_states"], Float[np.ndarray, "num_tokens num_states"]]:
    ground_truth_belief_states, activations = generate_belief_state_and_activations(model=model, process=process, num_sequences=num_sequences)

    # Reshape activations
    ground_truth_belief_states_reshaped = ground_truth_belief_states.reshape(-1, ground_truth_belief_states.shape[-1])
    activations_reshaped = activations.reshape(-1, activations.shape[-1])
    
    # Run Linear Regression
    reg = LinearRegression().fit(activations_reshaped, ground_truth_belief_states_reshaped)
    predicted_beliefs = reg.predict(activations_reshaped)
    
    return ground_truth_belief_states_reshaped, predicted_beliefs

def get_activations(model, nn_inputs, nn_type):
    if nn_type == 'transformer':
        _, cache = model.run_with_cache(nn_inputs, names_filter=lambda x: 'resid' in x or 'ln_final.hook_normalized' in x)
        max_layers = 10
        relevant_activation_keys = ['blocks.0.hook_resid_pre'] + [f'blocks.{i}.hook_resid_post' for i in range(max_layers)] + ['ln_final.hook_normalized']
        acts = torch.stack([v for k,v in cache.items() if k in relevant_activation_keys and k in cache], dim=0)
        return  acts
    elif nn_type == 'rnn':
        a, b = model.forward_with_all_states(nn_inputs)
        return b['layer_states']
    else:
        raise ValueError(f"Model type {nn_type} not supported")

def save_figure_to_s3(loader: S3ModelLoader, fig, sweep_id: str, run_id: str, checkpoint_key: str, title: str):
    """
    Save matplotlib figure to S3.
    
    Args:
        loader: S3ModelLoader instance
        fig: matplotlib figure object
        sweep_id: ID of the sweep
        run_id: ID of the run
        checkpoint_key: Key of the checkpoint
        title: Title/name for the figure
    """
    # Extract checkpoint number
    checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
    
    # Create a buffer to store the image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    
    # Construct the analysis path (sanitize title for use in filename)
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    analysis_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/figures/{safe_title}.png"
    
    # Upload to S3
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=analysis_key,
        Body=buf.getvalue(),
        ContentType='image/png'
    )

def plot_belief_prediction_comparison(
    nn_beliefs, nn_belief_indices, 
    belief_predictions, belief_predictions_shuffled, belief_predictions_cv,
    sweep_type, mse, mse_shuffled, mse_cv, test_inds,
    run_id, title=None, loader=None, checkpoint_key=None, sweep_id=None
):
    """Plot and optionally save belief prediction comparisons."""
    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    
    plot_belief_predictions2(
        belief_predictions=nn_beliefs,
        transformer_input_beliefs=nn_beliefs,
        transformer_input_belief_indices=nn_belief_indices,
        ax=ax[0],
        type=sweep_type,
        title=f'ground truth',
        mode='true',
        include_colorbar=False,
    )
    plot_belief_predictions2(
        belief_predictions=belief_predictions,
        transformer_input_beliefs=nn_beliefs,
        transformer_input_belief_indices=nn_belief_indices,
        ax=ax[1],
        type=sweep_type,
        title=f'MSE: {mse:.4E}',
        mode='predicted',
        include_colorbar=False,
    )

    plot_belief_predictions2(
        belief_predictions=belief_predictions_shuffled,
        transformer_input_beliefs=nn_beliefs,
        transformer_input_belief_indices=nn_belief_indices,
        ax=ax[2],
        type=sweep_type,
        title=f'MSE Shuffled: {mse_shuffled:.4E}',
        mode='predicted',
        include_colorbar=False,
    )

    plot_belief_predictions2(
        belief_predictions=belief_predictions_cv,
        transformer_input_beliefs=nn_beliefs,
        transformer_input_belief_indices=nn_belief_indices,
        ax=ax[3],
        type=sweep_type,
        title=f'MSE CV: {mse_cv:.4E}',
        include_colorbar=False,
        cv_test_inds=test_inds,
        mode='predicted',
    )

    # make sure all subplots have the same x and y lims
    for i in range(4):
        ax[i].set_xlim(ax[0].get_xlim())
        ax[i].set_ylim(ax[0].get_ylim())

    # no legend
    for i in range(4):
        ax[i].legend().set_visible(False)

    # add run_id as global title
    fig.suptitle(title)
    plt.tight_layout()
    
    # Save figure if requested
    if loader is not None and checkpoint_key is not None:
        save_figure_to_s3(
            loader=loader,
            fig=fig,
            sweep_id=sweep_id,
            run_id=run_id,
            checkpoint_key=checkpoint_key,
            title=f"belief_predictions_{title}" if title else "belief_predictions"
        )
    
    plt.show()
    plt.close()

def analyze_layer(layer_acts, nn_beliefs, nn_belief_indices, nn_probs, 
                 sweep_type, run_name, layer_idx, title=None, return_results=False,
                 loader=None, checkpoint_key=None, sweep_id=None, run_id=None, save_figure=False):  # Added save_figure parameter
    """Analyze a single layer's activations and plot results."""
    #print(f"Layer {layer_idx} shape:", layer_acts.shape)
    
    (regression, belief_predictions, mse, mse_shuffled, 
     belief_predictions_shuffled, mse_cv, belief_predictions_cv, 
     test_inds) = run_activation_to_beliefs_regression(
        layer_acts, nn_beliefs, nn_probs
    )
    if save_figure:  # Use the parameter
        plot_belief_prediction_comparison(
            nn_beliefs, nn_belief_indices, 
            belief_predictions, belief_predictions_shuffled, belief_predictions_cv,
            sweep_type, mse, mse_shuffled, mse_cv, test_inds,
            run_name,
            title=title,
            loader=loader,
            checkpoint_key=checkpoint_key,
            sweep_id=sweep_id,
            save_figure=save_figure
        )

    if return_results:
        return {
            'mse': float(mse),
            'mse_shuffled': float(mse_shuffled),
            'mse_cv': float(mse_cv),
            'regression_coef': regression.coef_,
            'regression_intercept': regression.intercept_,
            'predictions': belief_predictions,
            'predictions_shuffled': belief_predictions_shuffled,
            'predictions_cv': belief_predictions_cv,
            'test_indices': test_inds
        }

def analyze_all_layers(acts, nn_beliefs, nn_belief_indices, nn_probs, 
                      sweep_type, run_name, title=None, return_results=False,
                      loader=None, checkpoint_key=None, sweep_id=None, run_id=None, save_figure=False):
    """Analyze concatenated activations from all layers."""
    all_layers_acts = acts.permute(1,2,0,3).reshape(acts.shape[1], acts.shape[2], -1)
    #print("All layers concatenated shape:", all_layers_acts.shape)

    (regression, belief_predictions, mse, mse_shuffled, 
     belief_predictions_shuffled, mse_cv, belief_predictions_cv, 
     test_inds) = run_activation_to_beliefs_regression(
        all_layers_acts, nn_beliefs, nn_probs
    )

    if save_figure:
        plot_belief_prediction_comparison(
            nn_beliefs, nn_belief_indices, 
            belief_predictions, belief_predictions_shuffled, belief_predictions_cv,
            sweep_type, mse, mse_shuffled, mse_cv, test_inds,
            run_name,
            title=title,
            loader=loader,
            checkpoint_key=checkpoint_key,
            sweep_id=sweep_id,
        )   

    if return_results:
        return {
            'mse': float(mse),
            'mse_shuffled': float(mse_shuffled),
            'mse_cv': float(mse_cv),
            'regression_coef': regression.coef_,
            'regression_intercept': regression.intercept_,
            'predictions': belief_predictions,
            'predictions_shuffled': belief_predictions_shuffled,
            'predictions_cv': belief_predictions_cv,
            'test_indices': test_inds
        }

def save_analysis_results(loader: S3ModelLoader, sweep_id: str, run_id: str, checkpoint_key: str, results: dict, title: str):
    """
    Save analysis results to S3 in an organized structure.
    
    Args:
        loader: S3ModelLoader instance
        sweep_id: ID of the sweep being analyzed (e.g., '20241121152808')
        run_id: ID of the run being analyzed
        checkpoint_key: Key of the checkpoint being analyzed
        results: Dictionary containing analysis results
        title: Title of the analysis, name to save as
    """
    # Extract checkpoint number from key (e.g., "sweep/run/1000.pt" -> "1000")
    checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
    
    # Construct the analysis path
    analysis_key = f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/results_{title}.json"
    
    # Convert results to JSON
    results_json = json.dumps(results, cls=NumpyEncoder)
    
    # Upload to S3
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=analysis_key,
        Body=results_json
    )

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy and torch types"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def analyze_model_checkpoint(model, nn_inputs, nn_type, nn_beliefs, nn_belief_indices, 
                           nn_probs, sweep_type, run_name, sweep_id, title=None, save_results=True, 
                           loader=None, checkpoint_key=None, save_figure=False):
    """Analyze a single model checkpoint and optionally save results."""
    
    print(f"Analyzing {checkpoint_key} - title {title}")
    acts = get_activations(model, nn_inputs, nn_type)
    results = {
        'model_type': nn_type,
        'sweep_type': sweep_type,
        'run_name': run_name,
        'title': title,
        'layers': []
    }

    if nn_type == 'transformer':
        layer_names = ['embed'] + [f'layer {i}' for i in range(1, acts.shape[0]-1)] + ['final norm']
    elif nn_type == 'rnn':
        layer_names = [f'layer {i}' for i in range(acts.shape[0])]
    else:
        raise ValueError(f"Model type {nn_type} not supported")

    # For RNNs with 1 layer, only analyze all layers together
    if nn_type == 'rnn' and acts.shape[0] == 1:
        title_all_layers = f"All Layers" + (f" - {title}")
        layer_results = analyze_all_layers(acts, nn_beliefs, nn_belief_indices, nn_probs,
                                         sweep_type, run_name, title_all_layers, return_results=True,
                                         loader=loader, checkpoint_key=checkpoint_key, sweep_id=sweep_id, run_id=run_name, save_figure=save_figure)
        results['all_layers'] = layer_results
    else:
        # Analyze each layer individually
        for layer_idx in range(acts.shape[0]):
            title_layer = f"Layer {layer_idx}" + (f" - {title}")
            layer_results = analyze_layer(acts[layer_idx], nn_beliefs, nn_belief_indices,
                                        nn_probs, sweep_type, run_name, layer_names[layer_idx], 
                                        title_layer, return_results=True,
                                        loader=loader, checkpoint_key=checkpoint_key, sweep_id=sweep_id, run_id=run_name, save_figure=save_figure)
            results['layers'].append({
                'layer_name': layer_names[layer_idx],
                **layer_results
            })
        
        # Analyze all layers together
        title_all_layers = f"All Layers" + (f" - {title}")
        all_layers_results = analyze_all_layers(acts, nn_beliefs, nn_belief_indices, nn_probs,
                                              sweep_type, run_name, title_all_layers, return_results=True,
                                              loader=loader, checkpoint_key=checkpoint_key, sweep_id=sweep_id, run_id=run_name)
        results['all_layers'] = all_layers_results

    if save_results and loader is not None and checkpoint_key is not None:
        save_analysis_results(loader, 
                            sweep_id=sweep_id, 
                            run_id=run_name, 
                            checkpoint_key=checkpoint_key, 
                            results=results,
                            title=title
                            )
    
    return results

def shuffle_belief_norms(unnormalized_beliefs):
    """
    Shuffle the norms of the beliefs.

    Args:
        unnormalized_beliefs: torch.Tensor [n_samples, n_context, num_states]
    
    Returns:
        torch.Tensor with same shape as input but shuffled norms
    """
    # Create a copy to avoid modifying the original
    beliefs = unnormalized_beliefs.clone()
    
    # Reshape to 2D
    beliefs_2d = beliefs.reshape(-1, beliefs.shape[-1])  # [n_samples * n_context, num_states]
    
    # Calculate norms (keeping on same device as input)
    norms = torch.norm(beliefs_2d, dim=1)  # [n_samples * n_context]
    
    # Shuffle the norms
    shuffled_norms = norms[torch.randperm(norms.size(0))]

    # apply the norms to beliefs_2d
    beliefs_2d_unit = beliefs_2d / norms.reshape(-1, 1)
    beliefs_2d = beliefs_2d_unit * shuffled_norms.reshape(-1, 1)
    
    return beliefs_2d.reshape(unnormalized_beliefs.shape)

def save_nn_data(loader: S3ModelLoader, sweep_id: str, run_id: str, nn_data: dict):
    """
    Save neural network data to S3 with optimizations for large tensors.
    """
    # Convert tensors to numpy arrays and reduce precision
    # Convert tensors to numpy arrays and reduce precision
    serializable_data = {}
    for key, value in nn_data.items():
        if isinstance(value, torch.Tensor):
            # Convert to float32 to reduce size and keep reasonable precision
            np_array = value.cpu().numpy().astype(np.float32)
            serializable_data[key] = np_array
        else:
            serializable_data[key] = value
    
    # Use numpy's save function instead of JSON for better efficiency with numerical data
    buf = io.BytesIO()
    np.savez_compressed(buf, **serializable_data)
    buf.seek(0)
    
    # Construct the analysis path with .npz extension
    analysis_key = f"analysis/{sweep_id}/{run_id}/nn_data.npz"
    
    # Upload to S3
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=analysis_key,
        Body=buf.getvalue()
    )
