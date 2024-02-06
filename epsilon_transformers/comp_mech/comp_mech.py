from .HMM import HMM
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from .Mixed_State_Tree import Mixed_State_Tree

def compute_next_distribution(
    epsilon_machine: np.ndarray, X_current: np.ndarray, output: int
) -> np.ndarray:
    """
    Compute the next mixed state distribution for a given output.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
    X_current (np.ndarray): The current mixed state distribution.
    output (int): The output symbol.

    Returns:
    np.ndarray: The next mixed state distribution.
    """
    X_next = np.einsum("sd, s -> d", epsilon_machine[output], X_current)
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next



def generate_sequences(
    hmm: HMM, num_sequences: int, sequence_length: int, return_states: bool = False
) -> np.ndarray:
    """
    Generate sequences from the given HMM.

    Parameters:
    hmm (HMM): The HMM to generate sequences from.
    num_sequences (int): The number of sequences to generate.
    sequence_length (int): The length of each sequence.
    return_states (bool): Whether to return the states as well as the emissions.

    Returns:
        np.ndarray: The generated sequences. If return_states is True, the states are also returned.
    """

    # get the transition probabilities, stationary distribution, and number of states
    T = hmm.transition_probs
    ss = hmm.stationary_distribution
    num_states = T.shape[1]

    # choose the initial state for each sequence
    initial_state = np.random.choice(num_states, num_sequences, p=ss, replace=True)

    # initialize the sequences
    emissions = np.zeros((num_sequences, sequence_length), dtype=int)
    next_states = np.zeros((num_sequences, sequence_length), dtype=int)

    # generate the sequences
    for i in range(num_sequences):
        state = initial_state[i]
        for j in range(sequence_length):
            # get the emission probabilities for the current state
            probs = T[:, state, :]  # emission, to state
            probs_flat = probs.flatten()

            # choose the next state according to probs
            choice_ind_flat = np.random.choice(len(probs_flat), 1, p=probs_flat)
            choice_ind = np.unravel_index(choice_ind_flat, probs.shape)
            emission = choice_ind[0][0]
            to_state = choice_ind[1][0]
            emissions[i, j] = emission
            next_states[i, j] = to_state
            state = to_state

    if return_states:
        # append initial states to left of next_states
        next_states = np.concatenate([initial_state[:, None], next_states], axis=1)
        return emissions, next_states
    else:
        return emissions


def compute_emission_probabilities(hmm: HMM, mixed_state: np.ndarray) -> np.ndarray:
    """
    Compute the probabilities associated with each emission given the current mixed state.

    Parameters:
    hmm (HMM): The HMM to compute emission probabilities for.
    mixed_state (np.ndarray): The mixed state to compute emission probabilities for.

    Returns:
        np.ndarray: The emission probabilities.
    """

    T = hmm.transition_probs
    emission_probs = np.einsum("s,esd->ed", mixed_state, T).sum(axis=1)
    emission_probs /= emission_probs.sum()
    return emission_probs


def explore_mixed_state_tree(
    hmm: HMM,
    mixed_state: np.ndarray,
    depth: int,
    current_path: Tuple[int] = (),
    current_depth: int = 0,
    path_prob: float = 1.0,
    emit_prob: float = 1.0,
) -> Mixed_State_Tree:
    current_mixed_state = Mixed_State_Tree(
        mixed_state, path_prob, emit_prob, current_path
    )

    if current_depth < depth:
        T = hmm.transition_probs
        num_emissions = T.shape[0]

        emission_probs = compute_emission_probabilities(hmm, mixed_state)

        for emission in range(num_emissions):
            emission_prob = emission_probs[emission]
            if emission_prob > 0:
                next_mixed_state_vector = compute_next_distribution(
                    T, mixed_state, emission
                )
                next_path = current_path + (emission,)
                next_path_prob = path_prob * emission_prob

                child_mixed_state = explore_mixed_state_tree(
                    hmm,
                    next_mixed_state_vector,
                    depth,
                    next_path,
                    current_depth + 1,
                    next_path_prob,
                    emission_prob,
                )
                current_mixed_state.add_child(child_mixed_state)

    return current_mixed_state


def mixed_state_tree(hmm: HMM, tree_depth: int = 10) -> Mixed_State_Tree:
    mixed_state = hmm.stationary_distribution
    return explore_mixed_state_tree(hmm, mixed_state, tree_depth)


def entropy_by_path_length(tree: Mixed_State_Tree) -> List[float]:
    """
    Finds all path probabilities associated with paths of length N for N from 0 to max,
    and computes the entropy of that distribution at every N.
    """
    # Determine the maximum depth of the tree
    max_depth = tree.max_depth()

    entropies = []
    for depth in range(max_depth + 1):
        path_probs = collect_path_probs(tree, depth)
        entropy = compute_entropy(path_probs)
        entropies.append(entropy)

    return entropies


def compute_entropy(probs: List[float]) -> float:
    """
    Compute the entropy of a distribution.
    """
    return -sum(p * np.log(p) for p in probs if p > 0)


def collect_path_probs(
    tree: Mixed_State_Tree, target_depth: int, return_paths: bool = False
) -> Union[List[float], List[Tuple[Tuple[int], float]]]:
    """
    Collects all path probabilities at a specific target depth. Optionally returns paths along with probabilities.

    :param tree: The root of the Mixed_State_Tree.
    :param target_depth: The target depth to collect path probabilities from.
    :param return_paths: If True, returns a list of tuples with each path and its probability. Otherwise, returns a list of probabilities.
    :return: List of probabilities or list of tuples (path, probability) depending on return_paths.
    """
    results = []

    def traverse(node: Mixed_State_Tree, current_depth: int):
        if current_depth == target_depth:
            if return_paths:
                results.append((node.path, node.path_prob))
            else:
                results.append(node.path_prob)
        for child in node.children:
            traverse(child, current_depth + 1)

    traverse(tree, 0)

    return results


def collect_emit_probs(tree: Mixed_State_Tree, target_depth: int) -> List[float]:
    """
    Collects all emission probabilities at a specific target depth.
    """
    emit_probs = []

    def traverse(node: Mixed_State_Tree, current_depth: int):
        if current_depth == target_depth:
            emit_probs.append(node.emit_prob)
        for child in node.children:
            traverse(child, current_depth + 1)

    traverse(tree, 0)
    return emit_probs
