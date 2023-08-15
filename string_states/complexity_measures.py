import numpy as np
from numpy import linalg as LA
from typing import Tuple, Dict
from finite_state_machine import FiniteStateMachine
from collections import deque, defaultdict
from numpy.linalg import eig

def calculate_joint_prob_dist(fsm: FiniteStateMachine) -> np.ndarray:
    num_states = len(fsm.states)
    joint_prob_dist = np.zeros((num_states, 2))

    steady_state_distribution = calculate_steady_state_distribution(fsm)
    

    for i, state in enumerate(fsm.states):
        if fsm.transition_exists(state, 0):
            joint_prob_dist[i, 0] = steady_state_distribution[i] * fsm.get_emission_prob(state, 0)
        if fsm.transition_exists(state, 1):
            joint_prob_dist[i, 1] = steady_state_distribution[i] * (1 - fsm.get_emission_prob(state, 0))

    return joint_prob_dist

def calculate_steady_state_distribution(fsm: FiniteStateMachine) -> np.ndarray:
    """
    Calculates the steady state distribution for a given Finite State Machine (FSM).

    The steady state distribution is a 1D array where each element `i` represents the probability of being in state `i`.

    Parameters
    ----------
    fsm : FiniteStateMachine
        The Finite State Machine for which the steady state distribution is to be calculated.

    Returns
    -------
    steady_state_distribution : np.ndarray
        The steady state distribution of the FSM. It is a 1D array with length equal to the number of states in the FSM.

    Notes
    -----
    This function assumes that the transition matrix of the FSM has been calculated.
    """

    eigenvalues, eigenvectors = LA.eig(fsm.transition_matrix)
    steady_state_indices = np.abs(eigenvalues - 1) < 1e-10
    steady_state_distribution = eigenvectors[:, steady_state_indices]
    steady_state_distribution /= np.sum(steady_state_distribution)

    # clean up complex values, sometimes we get a vanishingly small imaginary part
    def deal_with_complex_part(value):
        if np.iscomplex(value):
            complex_part = np.imag(value)
            if complex_part < 1e-10:
                return np.squeeze(np.real(value))
            else:
                raise ValueError("value is complex")
        else:
            return np.squeeze(np.real(value))
        
    steady_state_distribution = np.array([deal_with_complex_part(value) for value in steady_state_distribution])


    return steady_state_distribution

def calculate_entropy_and_complexity(fsm: FiniteStateMachine) -> Tuple[float, float]:
    steady_state_distribution = calculate_steady_state_distribution(fsm)

    joint_prob_dist = calculate_joint_prob_dist(fsm)
    conditional_prob_dist = joint_prob_dist / np.sum(joint_prob_dist, axis=1, keepdims=True)

    # go through every state
    entropies = np.zeros_like(fsm.states, dtype=np.float64)
    for i in range(len(fsm.states)):
        # get the prob distribution over outputs
        prob_dist = conditional_prob_dist[i]
        
        # calculate the entropy of the prob dist, be sure to exclude zero probabilities
        nonzero_prob_dist = prob_dist[prob_dist > 0]
        entropies[i] = -np.sum(nonzero_prob_dist * np.log2(nonzero_prob_dist))

    entropy_rate = np.dot(entropies, steady_state_distribution)

    # statistical complexity is just the entropy of the steady state distribution
    statistical_complexity = -np.nansum(steady_state_distribution * np.log2(steady_state_distribution))

    def deal_with_complex_part(value):
        if np.iscomplex(value):
            complex_part = np.imag(value)
            if complex_part < 1e-10:
                return np.squeeze(np.real(value))
            else:
                raise ValueError("value is complex")
        else:
            return np.squeeze(np.real(value))
        
    return deal_with_complex_part(entropy_rate), deal_with_complex_part(statistical_complexity)

"""
def calculate_entropy_and_complexity(fsm: FiniteStateMachine) -> Tuple[float, float]:
    # Calculate the steady state distribution
    steady_state_distribution = calculate_steady_state_distribution(fsm)

    # Calculate p0 and p1
    p0 = 0
    accuracy = 0
    for i in range(len(fsm.states)):
        accuracy += steady_state_distribution[i] * np.max([fsm.emmision_0_probs[i], 1 - fsm.emmision_0_probs[i]])
        if fsm.emmision_0_probs[i] > 0.5:
            p0 += steady_state_distribution[i]
    p1 = 1 - p0

    # Calculate entropy rate
    entropy = -p0 * np.log(p0) if p0 != 0 else 0
    entropy -= p1 * np.log(p1) if p1 != 0 else 0
    entropy_rate = -np.nansum(steady_state_distribution * np.log(steady_state_distribution))


    # Calculate statistical complexity
    complexities = np.zeros_like(fsm.emmision_0_probs)
    mask = fsm.emmision_0_probs != 0
    complexities[mask] = -fsm.emmision_0_probs[mask] * np.log(fsm.emmision_0_probs[mask])
    mask = fsm.emmision_0_probs != 1
    complexities[mask] -= (1 - fsm.emmision_0_probs[mask]) * np.log(1 - fsm.emmision_0_probs[mask])
    statistical_complexity = np.dot(complexities, steady_state_distribution)

    def deal_with_complex_part(value):
        if np.iscomplex(value):
            complex_part = np.imag(value)
            if complex_part < 1e-10:
                return np.squeeze(np.real(value))
            else:
                raise ValueError("value is complex")
        else:
            return np.squeeze(np.real(value))
                
    entropy_rate = deal_with_complex_part(entropy_rate)
    statistical_complexity = deal_with_complex_part(statistical_complexity)

    return entropy_rate, statistical_complexity
"""


def rate_distortion(p, beta, tol=1e-6, max_iter=5000):
    """
    Calculate the rate-distortion function for a given joint probability distribution p and a beta value.

    Args:
    p: Joint probability distribution.
    beta: Beta value, which can be interpreted as the inverse temperature in statistical physics.
    tol: Tolerance for convergence. If the change in marginal_pXhat and conditional_pXhat_given_S is less than tol,
         the function will stop iterating and return the current rate and distortion.
    max_iter: Maximum number of iterations.

    Returns:
    R: The rate value in the rate-distortion function.
    D: The distortion value in the rate-distortion function.
    """
    
    # Calculate the marginal probabilities of X and S
    marginal_pX = np.sum(p, 0)
    marginal_pS = np.sum(p, 1)

    # Calculate conditional probability of X given S
    conditional_pX_given_S = np.dot(np.diag(1 / marginal_pS), p)

    # Distortion matrix is equal to the conditional probability of X given S
    distortion_matrix = conditional_pX_given_S

    # Initialize the conditional probability of Xhat given S randomly
    initial_conditional_pXhat_given_S0 = np.random.uniform(size=len(marginal_pS))
    conditional_pXhat_given_S = np.vstack([initial_conditional_pXhat_given_S0, 1 - initial_conditional_pXhat_given_S0]).T

    # Calculate the marginal probability of Xhat
    marginal_pXhat = np.dot(conditional_pXhat_given_S.T, marginal_pS)

    # Iterate to refine the conditional probability of Xhat given S and marginal probability of Xhat
    for _ in range(max_iter):
        # Calculate new estimates
        log_conditional_pXhat_given_S = np.meshgrid(np.log2(marginal_pXhat), np.ones(len(marginal_pS)))[0] + beta * distortion_matrix
        new_conditional_pXhat_given_S = np.exp(log_conditional_pXhat_given_S)
        normalization_constants = np.sum(new_conditional_pXhat_given_S, 1)
        new_conditional_pXhat_given_S = np.dot(np.diag(1 / normalization_constants), new_conditional_pXhat_given_S)
        new_marginal_pXhat = np.dot(new_conditional_pXhat_given_S.T, marginal_pS)
        
        # Check for convergence
        if np.allclose(marginal_pXhat, new_marginal_pXhat, atol=tol) and np.allclose(conditional_pXhat_given_S, new_conditional_pXhat_given_S, atol=tol):
            # print('beta=', beta, 'converged after', _+1, 'iterations')
            break
        
        # Update estimates
        marginal_pXhat = new_marginal_pXhat
        conditional_pXhat_given_S = new_conditional_pXhat_given_S

    # Calculate the rate value R in the rate-distortion function
    R = -np.nansum(marginal_pXhat * np.log2(marginal_pXhat)) + np.dot(marginal_pS, np.nansum(conditional_pXhat_given_S * np.log2(conditional_pXhat_given_S), 1))

    # Calculate the distortion value D in the rate-distortion function
    D = np.dot(marginal_pS, np.sum(conditional_pXhat_given_S * conditional_pX_given_S, 1))

    return R, D

def empirical_entropy_and_accuracy(actual_values, predicted_values):
    probability_of_one = np.mean(predicted_values == 1)
    probability_of_zero = 1 - probability_of_one

    if probability_of_one in {0, 1}:
        entropy = 0
    else:
        entropy = -probability_of_one * np.log2(probability_of_one) - probability_of_zero * np.log2(probability_of_zero)

    #accuracy = np.mean((actual_values == 1) == (predicted_values == 1))
    accuracy = np.mean(actual_values == predicted_values)

    return entropy, accuracy


def calculate_sequence_probabilities(fsm: FiniteStateMachine, max_length: int):
    all_sequence_probs = defaultdict(dict)
    queue = deque()

    # Calculate the steady state distribution
    steady_state_distribution = calculate_steady_state_distribution(fsm)

    # Initialize queue considering the steady state distribution
    for i, state in enumerate(fsm.states):
        for output in [0, 1]:
            if str(state) + str(output) in fsm.transition_function:
                sequence = str(output)
                new_state = fsm.transition_function[str(state) + str(output)]
                transition_prob = fsm.emmision_0_probs[state] if output == 0 else 1 - fsm.emmision_0_probs[state]
                initial_prob = steady_state_distribution[i] * transition_prob
                queue.append((sequence, new_state, initial_prob))
                all_sequence_probs[1][sequence] = initial_prob

    # BFS
    while queue:
        sequence, state, prob = queue.popleft()
        if len(sequence) < max_length:
            for output in [0, 1]:
                if str(state) + str(output) in fsm.transition_function:
                    new_sequence = sequence + str(output)
                    new_state = fsm.transition_function[str(state) + str(output)]
                    transition_prob = fsm.emmision_0_probs[state] if output == 0 else 1 - fsm.emmision_0_probs[state]
                    new_prob = prob * transition_prob
                    queue.append((new_sequence, new_state, new_prob))
                    all_sequence_probs[len(new_sequence)][new_sequence] = new_prob

    return dict(all_sequence_probs)


def calculate_block_entropies(sequence_probs: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    """
    Calculate the entropy for each sequence length.

    Args:
    sequence_probs: A dictionary where keys are sequence lengths and values are dictionaries. 
    Each nested dictionary represents the probability distribution of sequences of the corresponding length.

    Returns:
    A dictionary where keys are sequence lengths and values are the entropy of the probability distribution 
    of sequences of the corresponding length.
    """
    entropies = {0:0}
    for length, probs in sequence_probs.items():
        entropies[length] = -np.sum([p * np.log2(p) for p in probs.values() if p > 0])

    return entropies