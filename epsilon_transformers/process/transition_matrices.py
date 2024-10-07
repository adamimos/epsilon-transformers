"""
This module contains functions for creating transition matrices for various processes.
"""
import numpy as np

def get_matrix_from_args(name: str, **kwargs):
    process_functions = {
        "post_quantum": post_quantum,
        "tom_quantum": tom_quantum,
        "fanizza": fanizza,
        "rrxor": rrxor,
        "mess3": mess3
    }
    
    if name in process_functions:
        return process_functions[name](**kwargs)
    else:
        raise ValueError(f"Invalid process name: {name}")

def post_quantum(alpha=np.exp(1), beta=0.5):
    """
    Creates a transition matrix for the Post Quantum Process.
    """
    # Validate conditions for alpha and beta
    if not (alpha > 1 > beta > 0):
        raise ValueError("Condition alpha > 1 > beta > 0 not satisfied")
    if alpha + beta == 2:
        raise ValueError("Condition alpha + beta ≠ 2 not satisfied")
    if np.isclose(np.log(alpha) / np.log(beta), np.round(np.log(alpha) / np.log(beta))):
        raise ValueError("Condition ln(alpha) / ln(beta) ∉ ℚ not satisfied")

    T = np.zeros((3, 3, 3))
    m0 = np.array([[1], [1], [0]])  # Column vector
    mu0 = np.array([[1, -1, -1]])  # Row vector
    
    def _intermediate_matrix(val):
        return np.array([
            [val, 0, 0],
            [0, 1, 0],
            [0, np.log(val), 1]
        ])
    
    T[0] = np.outer(m0, mu0)
    T[1] = _intermediate_matrix(alpha)
    T[2] = _intermediate_matrix(beta)

    # Normalize T such that T[0] + T[1] + T[2] has largest abs eigenvalue = 1
    T_sum = T.sum(axis=0)
    T_sum_max_eigval = np.abs(np.linalg.eigvals(T_sum)).max()
    T /= T_sum_max_eigval

    # Verify that T[0] + T[1] + T[2] has largest abs eigenvalue = 1
    T_sum_normalized = T.sum(axis=0)
    T_sum_max_eigval = np.abs(np.linalg.eigvals(T_sum_normalized)).max()
    np.testing.assert_almost_equal(T_sum_max_eigval, 1, decimal=10, 
                                   err_msg="Largest absolute eigenvalue is not 1")

    return T


def tom_quantum(alpha: float, beta: float):
    """
    Creates a transition matrix for the Tom Quantum Process.
    """
    # Create a 4x3x3 array filled with zeros
    T= np.zeros((4, 3, 3))
    
    # Common elements
    gamma = 1/(2*np.sqrt(alpha**2+beta**2))
    common_diag = 1/4
    middle_diag = (alpha**2 - beta**2) * gamma**2
    off_diag = 2 * alpha * beta * gamma**2
    
    # G^(0)
    T[0] = np.array([
        [common_diag, 0, off_diag],
        [0, middle_diag, 0],
        [off_diag, 0, common_diag]
    ])
    
    # G^(1)
    T[1] = np.array([
        [common_diag, 0, -off_diag],
        [0, middle_diag, 0],
        [-off_diag, 0, common_diag]
    ])
    
    # G^(2)
    T[2] = np.array([
        [common_diag, off_diag, 0],
        [off_diag, common_diag, 0],
        [0, 0, middle_diag]
    ])
    
    # G^(3)
    T[3] = np.array([
        [common_diag, -off_diag, 0],
        [-off_diag, common_diag, 0],
        [0, 0, middle_diag]
    ])
    
    return T


def fanizza(alpha: float, lamb: float):
    """
    Creates a transition matrix for the Faniza Process.
    """
    # Calculate intermediate values
    a_la = (1 - lamb * np.cos(alpha) + lamb * np.sin(alpha)) / (1 - 2 * lamb * np.cos(alpha) + lamb**2)
    b_la = (1 - lamb * np.cos(alpha) - lamb * np.sin(alpha)) / (1 - 2 * lamb * np.cos(alpha) + lamb**2)

    # Define tau
    tau = np.ones(4)

    # Define the reset distribution pi0
    pi0 = np.array([1 - (2 / (1 - lamb) - a_la - b_la) / 4, 1 / (2 * (1 - lamb)), -a_la / 4, -b_la / 4])

    # Define w
    w = np.array([1, 1 - lamb, 1 + lamb * (np.sin(alpha) - np.cos(alpha)), 1 - lamb * (np.sin(alpha) + np.cos(alpha))])

    # Create Da
    Da = np.outer(w, pi0)

    # Create Db (with the sine sign error fixed)
    Db = lamb * np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(alpha), -np.sin(alpha)],
        [0, 0, np.sin(alpha), np.cos(alpha)]
    ])

    # Create the transition matrix T
    T = np.zeros((2, 4, 4))
    T[0] = Da
    T[1] = Db

    # Verify that T @ tau = tau (stochasticity condition)
    assert np.allclose(T[0] @ tau + T[1] @ tau, tau), "Stochasticity condition not met"

    return T

def rrxor(pR1=0.5, pR2=0.5):
    """
    Creates a transition matrix for the RRXOR Process.
    """
    T = np.zeros((2, 5, 5))
    s = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
    T[0, s["S"], s["0"]] = pR1
    T[1, s["S"], s["1"]] = 1 - pR1
    T[0, s["0"], s["F"]] = pR2
    T[1, s["0"], s["T"]] = 1 - pR2
    T[0, s["1"], s["T"]] = pR2
    T[1, s["1"], s["F"]] = 1 - pR2
    T[1, s["T"], s["S"]] = 1.0
    T[0, s["F"], s["S"]] = 1.0

    return T

def mess3(x=0.15, a=0.6):
    """
    Creates a transition matrix for the Mess3 Process.
    """
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

    return T
