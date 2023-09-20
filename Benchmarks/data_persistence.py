import numpy as np
from typing import List, Tuple

def save_epsilon_machine_to_file(epsilon_machine: np.ndarray, num_states: int, repeat_index: int) -> str:
    """Save the epsilon machine to a file and return the filename."""
    filename = f"data/epsilon_machine_{num_states}_{repeat_index}.npz"
    np.savez(filename, epsilon_machine=epsilon_machine)
    return filename

 