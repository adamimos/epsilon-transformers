from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def nond() -> HMM:
    """
    defined by Marzen and Crutchfield in https://arxiv.org/pdf/1702.08565.pdf
    """

    T = np.zeros((2, 3, 3))
    T[0, 0, 2] = 1.0
    T[1, 1, 0] = 0.5
    T[1, 1, 1] = 0.5
    T[1, 2, :] = 1.0 / 3.0

    return HMM(T)
