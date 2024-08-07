from scipy.optimize import minimize_scalar  # type: ignore
import numpy as np

# TODO: Check w/ Adam on which of these to keep

NUM_SYMBOLS = 2


def compute_minimum_error(epsilon_machine: np.ndarray) -> float:
    raise NotImplementedError


def binary_entropy(p: float) -> float:
    """Compute the binary entropy for a given probability p."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def inverse_binary_entropy(target_entropy: float) -> float:
    """Find the probability p corresponding to a given binary entropy value."""
    # Objective function: the difference between target entropy and binary entropy of p
    objective = lambda p: (binary_entropy(p) - target_entropy) ** 2

    # Minimize the objective function to find p
    result = minimize_scalar(objective, bounds=(0, 0.5), method="bounded")

    return result.x
