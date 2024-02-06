from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def golden_mean(R: int, k: int, p: float) -> HMM:
    """
    get the Golden Mean HMM

    Parameters:
    R (int): The number of states that output 1.
    k (int): The number of states that output 0.
    p (float): The probability of outputting 1 in the final state.
    """

    assert k <= R, "k must be less than or equal to R"
    n_states = R + k
    state_names = {chr(65 + i): i for i in range(n_states)}  # chr(65) is 'A'

    T = np.zeros((2, n_states, n_states))

    # First state
    T[1, state_names["A"], state_names["B"]] = p
    T[0, state_names["A"], state_names["A"]] = 1 - p

    # States that output 1
    for i in range(1, R):
        T[1, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

    # States that output 0
    for i in range(R, R + k - 1):
        T[0, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

    # Last state
    T[0, state_names[chr(65 + n_states - 1)], state_names["A"]] = 1.0

    return HMM(T)
