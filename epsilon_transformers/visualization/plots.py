import numpy as np
from jaxtyping import Float

from epsilon_transformers.process.processes import ZeroOneR

def _project_to_simplex(points: Float[np.ndarray, "num_points num_states"]):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y
