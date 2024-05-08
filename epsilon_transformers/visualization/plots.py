from typing import Literal
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
from matplotlib.figure import Figure
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from jaxtyping import Float
from epsilon_transformers.analysis.activation_analysis import find_msp_subspace_in_residual_stream

from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.training.configs.model_configs import RawModelConfig

# TODO: TQDM plot_ground_truth_and_evaluated_2d_simplex
# TODO: Modularize generate_belief_state_figures_datashader && parallalize slow tensor code

def _project_to_simplex(points: Float[np.ndarray, "num_points num_states"]):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y

# Combine aggregated channels into RGB images
def _combine_channels_to_rgb(agg_r, agg_g, agg_b, px:int):
    img_r = tf.shade(agg_r, cmap=['black', 'red'], how='linear')
    img_g = tf.shade(agg_g, cmap=['black', 'green'], how='linear')
    img_b = tf.shade(agg_b, cmap=['black', 'blue'], how='linear')

    img_r = tf.spread(img_r, px=px, shape='circle')
    img_g = tf.spread(img_g, px=px, shape='circle')
    img_b = tf.spread(img_b, px=px, shape='circle')

    # Combine using numpy
    r_array = np.array(img_r.to_pil()).astype(np.float64)
    g_array = np.array(img_g.to_pil()).astype(np.float64)
    b_array = np.array(img_b.to_pil()).astype(np.float64)
    
    # Stack arrays into an RGB image (ignoring alpha channel for simplicity)
    rgb_image = np.stack([r_array[:,:,0], g_array[:,:,1], b_array[:,:,2]], axis=-1)
    
    return Image.fromarray(np.uint8(rgb_image))

# TODO: I changed up the code for this to something which makes sense to me (creating panda dataframes from ground truth and predicted simplex. Check to see if this is what should actually be done)
def plot_ground_truth_and_evaluated_2d_simplex(
    ground_truth_tensor: Float[np.ndarray, "num_points num_states"], 
    predicted_beliefs: Float[np.ndarray, "num_points num_states"], 
    plot_triangles: bool,
    facecolor: Literal['black', 'white'],
    px: int
) -> Figure:
    # Projection and DataFrame preparation
    bs_x, bs_y = _project_to_simplex(np.array(ground_truth_tensor))
    ground_truth_data_frame = pd.DataFrame({'x': bs_x, 'y': bs_y, 'r': ground_truth_tensor[:, 0], 'g': ground_truth_tensor[:, 1], 'b': ground_truth_tensor[:, 2]})

    pb_x, pb_y = _project_to_simplex(np.array(predicted_beliefs))
    predicted_belief_vector_data_frame = pd.DataFrame({'x': pb_x, 'y': pb_y, 'r': ground_truth_tensor[:, 0], 'g': ground_truth_tensor[:, 1], 'b': ground_truth_tensor[:, 2]})

    # Create canvas
    canvas = ds.Canvas(plot_width=3000, plot_height=3000, x_range=(-0.1, 1.1), y_range=(-0.1, np.sqrt(3)/2 + 0.1))
    
    # Aggregate each RGB channel separately for ground truth and predicted beliefs
    colours = ['r', 'g', 'b']
    ground_truth_aggregated = {color: canvas.points(ground_truth_data_frame, 'x', 'y', ds.mean(color)) for color in colours}
    predited_belief_vector_aggregated = {color: canvas.points(predicted_belief_vector_data_frame, 'x', 'y', ds.mean(color)) for color in colours}

    img_gt = _combine_channels_to_rgb(ground_truth_aggregated['r'], ground_truth_aggregated['g'], ground_truth_aggregated['b'], px=3*px)
    img_pb = _combine_channels_to_rgb(predited_belief_vector_aggregated['r'], predited_belief_vector_aggregated['g'], predited_belief_vector_aggregated['b'], px=px)

    # Visualization with Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, facecolor=facecolor)
    for ax in axs:
        ax.tick_params(axis='x', colors=facecolor)  
        ax.tick_params(axis='y', colors=facecolor)  
        ax.xaxis.label.set_color(facecolor)  
        ax.yaxis.label.set_color(facecolor)  
        ax.title.set_color(facecolor)
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

if __name__ == "__main__":
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=10,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    process = ZeroOneR()

    belief_states_reshaped, predicted_beliefs = find_msp_subspace_in_residual_stream(model=model, process=process, num_sequences=5)
    plot_ground_truth_and_evaluated_2d_simplex(ground_truth_tensor=belief_states_reshaped, predicted_beliefs=predicted_beliefs, plot_triangles=True)