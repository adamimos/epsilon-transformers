from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def mess3(x: float = 0.15, a: float = 0.6) -> HMM:
    """
    defined by Marzen and Crutchfield in https://arxiv.org/pdf/1702.08565.pdf
    """

    # initialize the transition tensor
    T = np.zeros((3, 3, 3))
    b = (1 - a) / 2
    y = 1 - 2 * x

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
    T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
    T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

    # [0] = T[0].T
    # T[1] = T[1].T
    # T[2] = T[2].T

    return HMM(T)
