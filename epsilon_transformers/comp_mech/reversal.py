'''
This module contains functions associated with reversal of processes.
'''

import numpy as np
from epsilon_transformers.process.processes import TransitionMatrixProcess
from epsilon_transformers.process.Process import Process

def _reverse_transition_matrix(transition_matrix: np.ndarray, steady_state_dist: np.ndarray = None) -> np.ndarray:
    '''
    Reverse and renormalize a transition matrix T.
    
    T_ijk = P(to_state_k, emit_i | from_state_j)
    
    Parameters:
    transition_matrix (numpy.ndarray): 3D transition matrix with shape (num_emissions, num_states, num_states)
    steady_state_dist (numpy.ndarray): Steady state distribution of the process. If not provided, it will be calculated from the transition matrix.
    
    Returns:   
    numpy.ndarray: Reversed and renormalized transition matrix
    '''

    # Step 1: Calculate the stationary distribution if not provided
    if steady_state_dist is None:
        T_combined = np.sum(transition_matrix, axis=0)  # Sum over all emissions
        eigenvalues, eigenvectors = np.linalg.eig(T_combined.T)
        stationary_dist = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        stationary_dist = stationary_dist / np.sum(stationary_dist)
    else:
        stationary_dist = steady_state_dist
    
    # Step 2: Transpose the matrix (reverse arrows) and apply stationary distribution
    T_reversed = np.transpose(transition_matrix, (0, 2, 1)) * stationary_dist
    
    # Step 3: Renormalize
    T_reversed /= np.sum(T_reversed, axis=(0, 2), keepdims=True)
    
    return T_reversed

def reverse_process(process: Process) -> TransitionMatrixProcess:
    transition_matrix_reverse = _reverse_transition_matrix(process.transition_matrix, process.steady_state_vector)
    reverse_state_names_dict = {name + "_R": idx for name, idx in process.state_names_dict.items()}
    reverse_process = TransitionMatrixProcess(transition_matrix=transition_matrix_reverse,
                                              state_names=reverse_state_names_dict,
                                              name=process.name + "_reverse")
    return reverse_process