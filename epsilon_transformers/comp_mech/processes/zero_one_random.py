from epsilon_transformers.comp_mech.HMM import HMM
import numpy as np


def zero_one_random(p: float = 0.5) -> HMM:
    # check that p is a valid probability
    if not 0 <= p <= 1:
        raise ValueError("p must be a valid probability.")

    # initialize the transition tensor
    T = zero_one_random_matrix(p)

    return HMM(T)


def zero_one_random_matrix(p: float = 0.5) -> np.ndarray:
    # check that p is a valid probability
    if not 0 <= p <= 1:
        raise ValueError("p must be a valid probability.")
    
    T = np.zeros((2, 3, 3))
    state_names = {"0": 0, "1": 1, "R": 2}
    T[0, state_names["0"], state_names["1"]] = 1.0
    T[1, state_names["1"], state_names["R"]] = 1.0
    T[0, state_names["R"], state_names["0"]] = p
    T[1, state_names["R"], state_names["0"]] = 1 - p

    return T
    

def zero_one_random_set(p: float = 0.5, n: int = 2) -> HMM:
    # check that p is a valid probability
    if not 0 <= p <= 1:
        raise ValueError("p must be a valid probability.")
    if not 0 < n:
        raise ValueError("n must be a positive integer.")

    # for each n we need to make a matrix of form
    matrices = [zero_one_random_matrix(p) for _ in range(n)]
    # put these matrices together on the diagonal
    # each matrix is 2x3x3
    # so the final matrix will be 2n x 3n x 3n
    # we need to put the matrices on the diagonal

    T = np.zeros((2 * n, 3 * n, 3 * n))
    for i in range(n):
        T[2 * i:2 * i + 2, 3 * i:3 * i + 3, 3 * i:3 * i + 3] = matrices[i]

    return HMM(T)
    

def zero_one_random_abstracted() -> HMM:

    single_process = zero_one_random_matrix()
    n_states = single_process.shape[1]
    n_emissions = single_process.shape[0]

    T = np.zeros((2*n_emissions, n_states, n_states))
    T[0:n_emissions,:,:] = single_process/2.
    T[n_emissions:,:,:] = single_process/2.

    return HMM(T)


