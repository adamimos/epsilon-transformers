from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def random_random_xor(pR1: float = 0.5, pR2: float = 0.5) -> HMM:
    # check that p_r1 and p_r2 are valid probabilities
    if not 0 <= pR1 <= 1 or not 0 <= pR2 <= 1:
        raise ValueError("p_r1 and p_r2 must be valid probabilities.")

    T = np.zeros((2, 5, 5))
    state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
    T[0, state_names["S"], state_names["0"]] = pR1
    T[1, state_names["S"], state_names["1"]] = 1 - pR1
    T[0, state_names["0"], state_names["F"]] = pR2
    T[1, state_names["0"], state_names["T"]] = 1 - pR2
    T[0, state_names["1"], state_names["T"]] = pR2
    T[1, state_names["1"], state_names["F"]] = 1 - pR2
    T[1, state_names["T"], state_names["S"]] = 1.0
    T[0, state_names["F"], state_names["S"]] = 1.0

    return HMM(T)
