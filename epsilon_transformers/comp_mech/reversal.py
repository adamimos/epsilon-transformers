"""
This module contains functions associated with reversal of processes.
"""

from typing import Optional
import numpy as np
from epsilon_transformers.process.processes import TransitionMatrixProcess
from epsilon_transformers.process.Process import Process
from epsilon_transformers.process.MixedStateTree import MixedStateTree
from typing import Tuple, Dict

def _reverse_transition_matrix(
    transition_matrix: np.ndarray, 
    steady_state_dist: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reverse and renormalize a transition matrix T.
    
    T_ijk = P(to_state_k, emit_i | from_state_j)
    
    Args:
        transition_matrix (np.ndarray): 3D transition matrix with shape (num_emissions, num_states, num_states)
        steady_state_dist (Optional[np.ndarray]): Steady state distribution of the process. 
            If not provided, it will be calculated from the transition matrix.
    
    Returns:   
        np.ndarray: Reversed and renormalized transition matrix
    """


    if steady_state_dist is None:
        T_combined = np.sum(transition_matrix, axis=0)  # Sum over all emissions
        eigenvalues, eigenvectors = np.linalg.eig(T_combined.T)
        stationary_dist = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        stationary_dist = stationary_dist / np.sum(stationary_dist)
    else:
        stationary_dist = steady_state_dist

    T_reversed = np.einsum("j,eji,i->eij", stationary_dist, transition_matrix, 1./stationary_dist)
    print(T_reversed)
    return T_reversed

def reverse_process(process: Process) -> TransitionMatrixProcess:
    """
    Create a reversed version of the given process.

    Args:
        process (Process): The process to reverse.

    Returns:
        TransitionMatrixProcess: A new process with reversed transition matrix.
    """
    transition_matrix_reverse = _reverse_transition_matrix(
        process.transition_matrix, 
        process.steady_state_vector
    )
    reverse_state_names_dict = {f"{name}_R": idx for name, idx in process.state_names_dict.items()}
    reverse_process = TransitionMatrixProcess(
        transition_matrix=transition_matrix_reverse,
        state_names=reverse_state_names_dict,
        name=f"{process.name}_reverse"
    )
    return reverse_process


def get_recurrent_component_process_from_mstree(msp: MixedStateTree) -> Tuple[TransitionMatrixProcess, Dict[int, np.ndarray]]:
    """
    Extract the recurrent component of a Mixed State Presentation as a new process.
    
    Args:
        msp (MixedStateTree): The Mixed State Presentation tree.
    
    Returns:
        Tuple[TransitionMatrixProcess, Dict[int, np.ndarray]]: 
            - A new process representing the recurrent component
            - A dictionary mapping recurrent state indices to their corresponding distributions over original states
    """
    MSP_transition_matrix, belief_distributions = msp.build_msp_transition_matrix()
    MPS_process = TransitionMatrixProcess(transition_matrix=MSP_transition_matrix, name='_msp')
    
    steady_state_vector = MPS_process.steady_state_vector
    recurrent_indices = np.where(steady_state_vector > 0)[0]
    
    recurrent_transition_matrix = MPS_process.transition_matrix[:, recurrent_indices, :][:, :, recurrent_indices]
    
    recurrent_state_to_original_belief = {
        i: belief_distributions[idx] 
        for i, idx in enumerate(recurrent_indices)
    }
    
    recurrent_process = TransitionMatrixProcess(transition_matrix=recurrent_transition_matrix, name='_recurrent')
    recurrent_process.recurrent_state_to_original_belief = recurrent_state_to_original_belief
    
    return recurrent_process, recurrent_state_to_original_belief

def get_prob_forward_given_past(epsilon_machine_rev: Process):
    """
    Calculate the probability matrix of forward states given past states.

    Args:
        epsilon_machine_rev (Process): The epsilon machine. Output of get_recurrent_component_process_from_mstree.

    Returns:
        np.ndarray: The probability matrix of forward states given past states.
    """
    num_recurrent_states = len(epsilon_machine_rev.recurrent_state_to_original_belief)
    num_original_states = len(next(iter(epsilon_machine_rev.recurrent_state_to_original_belief.values())))
    
    prob_matrix = np.zeros((num_recurrent_states, num_original_states))
    for i, original_belief in epsilon_machine_rev.recurrent_state_to_original_belief.items():
        prob_matrix[i, :] = original_belief
    
    return prob_matrix.T


def joint_prob_from_conditional(conditional_prob: np.ndarray, marginal_prob: np.ndarray) -> np.ndarray:
    """
    Calculate the joint probability from the conditional probability and the marginal probability.

    conditional_prob (np.ndarray): Conditional probability matrix with shape (num_forward_states, num_reverse_states), p(f|r)
    marginal_prob (np.ndarray): Marginal probability vector with shape (num_reverse_states,), p(r), usually the steady state distribution
    """
    return np.einsum("ij,j->ij", conditional_prob, marginal_prob)