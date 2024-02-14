import numpy as np
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float

from epsilon_transformers.processes.rrxor import RRXOR
from epsilon_transformers.processes.mess3 import Mess3
from epsilon_transformers.processes.zero_one_random import ZeroOneR  

# TODO: Add derive_msp() to processes

PROCESS_REGISTRY: Dict[str, 'Process'] = {
    'z1r': ZeroOneR(),
    'mess3': Mess3(),
    'rrxor': RRXOR()
}

@dataclass
class ProcessHistory:
    symbols: List[int]
    states: List[str]

    def __post_init__(self):
        assert len(self.symbols) == len(self.states), 'length of symbols & states must be the same'

    def __len__(self):
        return len(self.states)

class Process(ABC):
    transition_matrix: Float[np.ndarray, 'vocab_len num_states num_states']
    state_names_dict: Dict[str, int]
    vocab_len: int
    num_states: int
    steady_state_vector: Float[np.ndarray, 'num_states']

    @property
    def steady_state_vector(self) -> Float[np.ndarray, 'num_states']:
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

        if len(self.transition_matrix.shape) != 3 or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should have 3 axes and the final two dims shoulds be square")

        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")
        
        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

    @abstractmethod
    def _create_hmm(self) -> Tuple[Float[np.ndarray, 'vocab_len num_states num_states'], Dict[str,int]]:
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...

    def _sample_emission(self, current_state_index: int) -> int:
        assert 0 <= current_state_index <= self.vocab_len, "current_state_index must be positive & less than vocab_len"
        
        p = self.transition_matrix[:, current_state_index, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        return emission

    def generate_process_history(self, total_length: int, current_state_idx: Optional[int] = None) -> ProcessHistory:
        """
        Generate a sequence of states based on the transition matrix.
        """        
        if current_state_idx is None:
            current_state_ind = np.random.choice(self.num_states, p=self.steady_state_vector)

        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        symbols = []
        states = []
        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_ind])
           
            emission = self._sample_emission(current_state_ind)
            symbols.append(emission)
           
            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(self.transition_matrix[emission, current_state_ind, :])
            current_state_ind = next_state_ind
        return ProcessHistory(symbols=symbols, states=states)