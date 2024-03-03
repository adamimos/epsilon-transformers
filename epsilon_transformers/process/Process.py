import numpy as np
from typing import Set, Tuple, Optional, Dict, List, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float
from collections import deque

from epsilon_transformers.process.MixedStatePresentation import MixedStateTree, MixedStateTreeNode

# TODO: Test yield_emission_histories for different emissions in the emission history
# TODO: Rename _create_hmm
# TODO: Delete generate_process_history (??)

@dataclass
class ProcessHistory:
    symbols: List[int]
    states: List[str]

    def __post_init__(self):
        assert len(self.symbols) == len(
            self.states
        ), "length of symbols & states must be the same"

    def __len__(self):
        return len(self.states)


class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: Dict[str, int]
    vocab_len: int
    num_states: int
    steady_state_vector: Float[np.ndarray, "num_states"]

    @property
    def steady_state_vector(self) -> Float[np.ndarray, "num_states"]:
        state_transition_matrix = np.sum(self.transition_matrix, axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(state_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        out: np.ndarray = normalized_steady_state_vector[:, 0]

        assert out.ndim == 1
        assert len(out) == self.num_states
        return out

    @property
    def is_unifilar(self) -> bool:
        # For each state, check if there are multiple transitions for each symbol
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                # If there are multiple transitions, return False
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    def __init__(self):
        self.transition_matrix, self.state_names_dict = self._create_hmm()

        if (
            len(self.transition_matrix.shape) != 3
            or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

    @abstractmethod
    def _create_hmm(
        self,
    ) -> Tuple[Float[np.ndarray, "vocab_len num_states num_states"], Dict[str, int]]:
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...

    def _sample_emission(self, current_state_idx: Optional[int] = None) -> int:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        p = self.transition_matrix[:, current_state_idx, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        return emission

    def yield_emissions(
        self, sequence_len: int, current_state_idx: Optional[int] = None
    ) -> Iterator[int]:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than vocab_len"

        for _ in range(sequence_len):
            emission = self._sample_emission(current_state_idx)
            yield emission

            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(
                self.transition_matrix[emission, current_state_idx, :]
            )
            current_state_idx = next_state_ind

    def yield_emission_histories(self, sequence_len: int, num_sequences: int) -> Iterator[List[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len)]

    def generate_process_history(
        self, total_length: int, current_state_idx: Optional[int] = None
    ) -> ProcessHistory:
        """
        Generate a sequence of states based on the transition matrix.
        """
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx <= self.vocab_len
        ), "current_state_index must be positive & less than vocab_len"

        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        symbols = []
        states = []
        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_idx])

            emission = self._sample_emission(current_state_idx)
            symbols.append(emission)

            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(
                self.transition_matrix[emission, current_state_idx, :]
            )
            current_state_idx = next_state_ind
        return ProcessHistory(symbols=symbols, states=states)
    
    # TODO: You can get rid of the stack, and just iterate through the nodes & the depth as tuples
    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        uniform_prior = np.full(self.num_states, 1/self.num_states)
        tree_root = MixedStateTreeNode(state_prob_vector=uniform_prior, children=set(), path=[])
        nodes = set([tree_root])

        stack = deque([(tree_root, uniform_prior, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(self, state_prob_vector)
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(self.transition_matrix, state_prob_vector, emission)
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(state_prob_vector=next_state_prob_vector, path=child_path, children=set())
                        current_node.add_child(child_node)

                        stack.append((child_node, next_state_prob_vector, child_path, current_depth + 1))
            nodes.add(current_node)
        
        return MixedStateTree(root_node=tree_root, process=self.name, nodes=nodes, depth=depth)

def _compute_emission_probabilities(
    hmm: Process, 
    state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
    """
    Compute the probabilities associated with each emission given the current mixed state.
    """
    T = hmm.transition_matrix
    emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
    emission_probs /= emission_probs.sum()
    return emission_probs

def _compute_next_distribution(
    epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
    current_state_prob_vector: Float[np.ndarray, "num_states"], 
    current_emission: int
) -> Float[np.ndarray, "num_states"]:
    """
    Compute the next mixed state distribution for a given output.
    """
    X_next = np.einsum("sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector)
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next