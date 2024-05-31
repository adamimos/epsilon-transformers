from typing import Literal
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
