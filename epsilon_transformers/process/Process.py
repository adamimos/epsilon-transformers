import numpy as np
from typing import Set, Tuple, Optional, Dict, List, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float
from collections import deque

from epsilon_transformers.process.MixedStateTree import MixedStateTree, MixedStateTreeNode

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

class GHMM(ABC):
    name: str
    transition_matrices: Float[np.ndarray, "vocab_len latent_dim latent_dim"]
    latent_dim: int
    vocab_len: int

    @property
    def steady_state_vector(self) -> Float[np.ndarray, "latent_dim"]:
        latent_transition_matrix = np.sum(self.transition_matrix, axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(latent_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real.T
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        out: np.ndarray = normalized_steady_state_vector

        assert out.ndim == 2
        assert out.shape == (1, self.latent_dim)
        return out

    @property
    def right_eigenvector(self) -> Float[np.ndarray, "latent_dim"]:
        latent_transition_matrix = np.sum(self.transition_matrices, axis=0)
        eigenvalues, eigenvectors = np.linalg.eig(latent_transition_matrix)
        right_eigenvector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        # Normalize the eigenvector so that the sum of components equals latent_dim
        right_eigenvector = right_eigenvector * (self.latent_dim / np.sum(right_eigenvector))
        assert right_eigenvector.ndim == 2
        assert right_eigenvector.shape == (self.latent_dim, 1)
        return right_eigenvector

    def __init__(self):
        self.transition_matrices, self.state_names_dict = self._create_ghmm()

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
        #if not np.allclose(transition.sum(axis=1), 1.0):
        #    raise ValueError("Transition matrix rows should sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.latent_dim = self.transition_matrix.shape[1]


        # TODO: add check on ssv (page 61 of Dan Upper's thesis)

    def word_probability(self, word: List[int]) -> float:
        """
        Compute the probability of a word using the GHMM.

        """
        T_w = np.identity(self.latent_dim)
        for symbol in word:
            T_w = T_w @ self.transition_matrices[symbol]
         
        word_probability = np.einsum("s,sd,d->", self.steady_state_vector, T_w, self.right_eigenvector)
        word_probability = word_probability/np.einsum("s,s->", self.steady_state_vector, self.right_eigenvector)
        return word_probability

    def _create_ghmm(self) -> Tuple[Float[np.ndarray, "vocab_len latent_dim latent_dim"], Dict[str, int]]:
        ...

    def _sample_emission_and_next_latent(self, latent_state: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
        if latent_state is None:
            latent_state = self.steady_state_vector
        else:
            assert isinstance(latent_state, np.ndarray) and latent_state.shape == (1,self.latent_dim), \
                "latent_state must be a row vector with length equal to latent_dim"
            
        ones = self.right_eigenvector
            
        # emission probs is (eta @ T @ ones) / (eta @ ones)
        emission_probs = latent_state @ self.transition_matrices @ ones
        emission_probs = emission_probs / (latent_state @ ones)

        # choose an emission
        emission = np.random.choice(self.vocab_len, p=emission_probs.squeeze())

        # next latent is (eta @ T[x]) / (eta @ T[x] @ ones)
        next_latent = (latent_state @ self.transition_matrices[emission])
        next_latent = next_latent / (next_latent @ ones)

        return emission, next_latent

    def yield_emissions(self, sequence_len: int, latent_state: Optional[np.ndarray] = None) -> Iterator[int]:
        if latent_state is None:
            latent_state = self.steady_state_vector
        else:
            assert isinstance(latent_state, np.ndarray) and latent_state.shape == (1, self.latent_dim), \
                "latent_state must be a row vector with length equal to latent_dim"

        for _ in range(sequence_len):
            emission, next_latent = self._sample_emission_and_next_latent(latent_state)
            yield emission
            latent_state = next_latent

    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        tree_root = MixedStateTreeNode(state_prob_vector=self.steady_state_vector, children=set(), path=[], emission_prob=0, path_prob=1.0)
        nodes = set([tree_root])

        stack = deque([(tree_root, self.steady_state_vector, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(self.transition_matrices, state_prob_vector, self.right_eigenvector)
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(
                            self.transition_matrices, 
                            state_prob_vector, 
                            emission,
                            self.right_eigenvector
                        )
                        # round to 5 decimal places
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector, 
                            path=child_path, 
                            children=set(), 
                            emission_prob=emission_probs[emission],
                            path_prob=current_node.path_prob * emission_probs[emission]
                        )
                        current_node.add_child(child_node)

                        stack.append((child_node, next_state_prob_vector, child_path, current_depth + 1))
            nodes.add(current_node)
        
        return MixedStateTree(root_node=tree_root, process=self.name, nodes=nodes, depth=depth)



class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: Dict[str, int]
    vocab_len: int
    num_states: int

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

    def __str__(self):
        return (f"{self.name} Process\n"
            f"Number of states: {self.num_states}\n"
            f"Vocabulary length: {self.vocab_len}\n"
            f"Transition matrix shape: {self.transition_matrix.shape}")
        

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

    def yield_emissions(self, sequence_len: int, current_state_idx: Optional[int] = None) -> Iterator[int]:
        if current_state_idx is None:
            current_state_idx = np.random.choice(self.num_states, p=self.steady_state_vector)
        assert 0 <= current_state_idx < self.num_states, "current_state_index must be positive & less than num_states"
        for _ in range(sequence_len):
            emission, next_state_idx = self._sample_emission_and_next_state(current_state_idx)
            yield emission
            current_state_idx = next_state_idx

    def _sample_emission_and_next_state(self, current_state_idx: int) -> Tuple[int, int]:
        transition_probs = self.transition_matrix[:, current_state_idx, :]
        emission_next_state_idx = np.random.choice(transition_probs.size, p=transition_probs.ravel())
        emission = emission_next_state_idx // self.num_states
        next_state_idx = emission_next_state_idx % self.num_states
        return emission, next_state_idx

    def yield_emission_histories(self, sequence_len: int, num_sequences: int) -> Iterator[List[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len)]

    def generate_process_history(self, total_length: int, current_state_idx: Optional[int] = None) -> ProcessHistory:
        if current_state_idx is None:
            current_state_idx = np.random.choice(self.num_states, p=self.steady_state_vector)
        assert 0 <= current_state_idx < self.num_states, "current_state_index must be positive & less than num_states"
        
        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}
        
        symbols = []
        states = []
        
        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_idx])
            emission, next_state_idx = self._sample_emission_and_next_state(current_state_idx)
            symbols.append(emission)
            current_state_idx = next_state_idx
            
        return ProcessHistory(symbols=symbols, states=states)
    
    # TODO: You can get rid of the stack, and just iterate through the nodes & the depth as tuples
    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        tree_root = MixedStateTreeNode(state_prob_vector=self.steady_state_vector, children=set(), path=[], emission_prob=0, path_prob=1.0)
        nodes = set([tree_root])

        stack = deque([(tree_root, self.steady_state_vector, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(self, state_prob_vector)
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(
                            self.transition_matrix, 
                            state_prob_vector, 
                            emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector, 
                            path=child_path, 
                            children=set(), 
                            emission_prob=emission_probs[emission],
                            path_prob=current_node.path_prob * emission_probs[emission]
                        )
                        current_node.add_child(child_node)

                        stack.append((child_node, next_state_prob_vector, child_path, current_depth + 1))
            nodes.add(current_node)
        
        return MixedStateTree(root_node=tree_root, process=self.name, nodes=nodes, depth=depth)

def _compute_emission_probabilities(
    transition_matrices: Float[np.ndarray, "vocab_len num_states num_states"],
    state_prob_vector: Float[np.ndarray, "1 num_states"],
    ones: Float[np.ndarray, "num_states 1"]
) -> Float[np.ndarray, "vocab_len"]:
    """
    Compute the probabilities associated with each emission given the current mixed state.
    """
    T = transition_matrices
    eta = state_prob_vector  # eta has shape (1, num_states)
    return ((eta @ T @ ones) / (eta @ ones)).squeeze() # shape (vocab_len)

def _compute_next_distribution(
    epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
    current_state_prob_vector: Float[np.ndarray, "1 num_states"], 
    current_emission: int,
    ones: Float[np.ndarray, "num_states 1"]
) -> Float[np.ndarray, "num_states"]:
    """
    Compute the next mixed state distribution for a given output.
    """
    T = epsilon_machine[current_emission] # shape (num_states, num_states)
    eta = current_state_prob_vector # shape (1, num_states)
    numerator = eta @ T # shape (1, num_states)
    denominator = numerator @ ones # shape (1)
    return numerator / denominator if denominator != 0 else numerator