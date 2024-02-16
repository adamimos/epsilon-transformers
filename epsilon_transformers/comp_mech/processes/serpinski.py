from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def serpinski(a: float = 1.0 / 6.0, s: float = 6.0) -> HMM:
    """
    defined byJurgens and Crutchfield in https://pubs.aip.org/aip/cha/article/31/8/083114/1059698
    equation D1
    """

    a_s = a * s
    one_minus_a_s = 1 - a_s
    one_minus_a_s_half = one_minus_a_s / 2
    a_s_minus_one_half_s = (one_minus_a_s * (s - 1)) / (2 * s)

    T = np.array(
        [
            [[a, 0, a * (s - 1)], [0, a, a * (s - 1)], [0, 0, a_s]],
            [
                [one_minus_a_s_half, 0, 0],
                [a_s_minus_one_half_s, one_minus_a_s_half, 0],
                [a_s_minus_one_half_s, 0, one_minus_a_s_half],
            ],
            [
                [one_minus_a_s_half, a_s_minus_one_half_s, 0],
                [0, one_minus_a_s_half, 0],
                [0, a_s_minus_one_half_s, a_s_minus_one_half_s],
            ],
        ]
    )
    # [0] = T[0].T
    # T[1] = T[1].T
    # T[2] = T[2].T

    return HMM(T)
