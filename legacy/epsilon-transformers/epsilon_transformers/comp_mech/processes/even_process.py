from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def even_process(p: float = 0.5) -> HMM:
    """
    This process only outputs even numbers of 1s
    """
    # check that p is a valid probability
    if not 0 <= p <= 1:
        raise ValueError("p must be a valid probability.")

    # initialize the transition tensor
    T = np.zeros((2, 2, 2))
    state_names = {"E": 0, "O": 1}
    T[1, state_names["E"], state_names["O"]] = 1 - p
    T[0, state_names["E"], state_names["E"]] = p
    T[1, state_names["O"], state_names["E"]] = 1.0

    return HMM(T)
