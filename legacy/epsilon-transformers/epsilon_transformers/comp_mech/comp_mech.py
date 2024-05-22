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


def generate_sequences_with_states(
    hmm: HMM, num_sequences: int, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sequences and their states from the given HMM.

    Parameters:
    hmm (HMM): The HMM to generate sequences from.
    num_sequences (int): The number of sequences to generate.
    sequence_length (int): The length of each sequence.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The generated sequences and their states.
    """
    emissions, next_states, initial_state = _generate_sequences_base(
        hmm, num_sequences, sequence_length
    )

    # append initial states to left of next_states
    next_states = np.concatenate([initial_state[:, None], next_states], axis=1)
    return emissions, next_states


def generate_sequences(
    hmm: HMM, num_sequences: int, sequence_length: int
) -> np.ndarray:
    """
    Generate sequences from the given HMM without returning the states.

    Parameters:
    hmm (HMM): The HMM to generate sequences from.
    num_sequences (int): The number of sequences to generate.
    sequence_length (int): The length of each sequence.

    Returns:
    np.ndarray: The generated sequences.
    """
    emissions, _, _ = _generate_sequences_base(hmm, num_sequences, sequence_length)
    return emissions


def _generate_sequences_base(
    hmm: HMM, num_sequences: int, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Base function to generate sequences and states from the given HMM.

    Parameters:
    hmm (HMM): The HMM to generate sequences from.
    num_sequences (int): The number of sequences to generate.
    sequence_length (int): The length of each sequence.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: The generated sequences, next states, and initial states.
    """
    T = hmm.transition_probs
    ss = hmm.stationary_distribution
    num_states = T.shape[1]

    initial_state = np.random.choice(num_states, num_sequences, p=ss, replace=True)

    emissions = np.zeros((num_sequences, sequence_length), dtype=int)
    next_states = np.zeros((num_sequences, sequence_length), dtype=int)

    for i in range(num_sequences):
        state = initial_state[i]
        for j in range(sequence_length):
            probs = T[:, state, :]
            probs_flat = probs.flatten()

            choice_ind_flat = np.random.choice(len(probs_flat), 1, p=probs_flat)
            choice_ind = np.unravel_index(choice_ind_flat, probs.shape)
            emission = choice_ind[0][0]
            to_state = choice_ind[1][0]

            emissions[i, j] = emission
            next_states[i, j] = to_state
            state = to_state

    return emissions, next_states, initial_state


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
    current_path: Tuple[int, ...] = (),
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


def entropy_by_path_length(
    tree: Mixed_State_Tree, max_length: Optional[int] = None
) -> List[float]:
    """
    Finds all path probabilities associated with paths of length N for N from 0 to max,
    and computes the entropy of that distribution at every N. If max_length is provided,
    computes entropies up to that length; otherwise, computes for all lengths.
    """
    # Determine the maximum depth of the tree
    max_depth = tree.max_depth()
    if max_length is not None:
        max_depth = min(max_depth, max_length + 1)

    entropies = []
    for depth in range(max_depth + 1):
        path_probs = collect_path_probs(tree, depth)
        if isinstance(path_probs, list) and all(
            isinstance(item, float) for item in path_probs
        ):
            entropy = compute_entropy(path_probs)
            entropies.append(entropy)
        else:
            raise ValueError("Path probabilities must be a list of floats.")

    return entropies


def compute_entropy(probs: List[float]) -> float:
    """
    Compute the entropy of a distribution.
    """
    return -sum(p * np.log(p) for p in probs if p > 0)


def collect_path_probs(tree: Mixed_State_Tree, target_depth: int) -> List[float]:
    """
    Collects all path probabilities at a specific target depth.

    :param tree: The root of the Mixed_State_Tree.
    :param target_depth: The target depth to collect path probabilities from.
    :return: List of probabilities.
    """
    results = []

    def traverse(node: Mixed_State_Tree, current_depth: int):
        if current_depth == target_depth:
            results.append(node.path_prob)
        for child in node.children:
            traverse(child, current_depth + 1)

    traverse(tree, 0)

    return results


def collect_path_probs_with_paths(
    tree: Mixed_State_Tree, target_depth: int
) -> List[Tuple[Tuple[int, ...], float]]:
    """
    Collects all path probabilities along with their paths at a specific target depth.

    :param tree: The root of the Mixed_State_Tree.
    :param target_depth: The target depth to collect path probabilities and their paths from.
    :return: List of tuples (path, probability).
    """
    results = []

    def traverse(node: Mixed_State_Tree, current_depth: int):
        if current_depth == target_depth:
            results.append((node.path, node.path_prob))
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


def block_entropy(
    tree: Mixed_State_Tree, max_depth: Optional[int] = None
) -> np.ndarray:
    """
    Compute the block entropy of the given mixed state tree.
    """
    return np.array(entropy_by_path_length(tree, max_depth))


def myopic_entropy(
    tree: Mixed_State_Tree, max_depth: Optional[int] = None
) -> np.ndarray:
    """
    Compute the myopic entropy of the given mixed state tree.
    """

    block_entropy = entropy_by_path_length(tree, max_depth)
    return np.diff(block_entropy)
