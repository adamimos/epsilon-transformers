from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def zero_one_random(p: float = 0.5) -> HMM:
    # check that p is a valid probability
    if not 0 <= p <= 1:
        raise ValueError("p must be a valid probability.")

    # initialize the transition tensor
    T = np.zeros((2, 3, 3))
    state_names = {"0": 0, "1": 1, "R": 2}
    T[0, state_names["0"], state_names["1"]] = 1.0
    T[1, state_names["1"], state_names["R"]] = 1.0
    T[0, state_names["R"], state_names["0"]] = p
    T[1, state_names["R"], state_names["0"]] = 1 - p

    return HMM(T)
