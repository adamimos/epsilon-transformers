import numpy as np
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float

from epsilon_transformers.markov_utilities import calculate_steady_state_distribution

# TODO: Bring up question in PR

# LUCAS Q: Why do we have to start our selection from the steady state distribution??
# Is it because this is what we expect the distribution to be at it's limit and we want to sample according to this limit?? If so is this the correct way of doing it??

# TODO: Test if steady state is calculated at init or if it's done lazily
# TODO: Test _sample_emission
# TODO: Test generate single sequence
# TODO: Test generate multiple sequences

# TODO: Write validator for ProcessHistory (same length)

# TODO: Add jaxtyping to steady_state 
# TODO: Clean up dead comments
# TODO: Check if is_unifilar is actually ever used

# TODO: Add Processes Registry
# TODO: Add derive_msp() to processes

@dataclass
class ProcessHistory:
    symbols: List[int]
    states: Optional[List[str]] = None

class Process(ABC):
    """
    This is the parent class for presentations of a process. It is responsible for initializing the transition matrix,
    calculating the steady state distribution, and setting the number of symbols and states.
    """
    transition_matrix: Float[np.ndarray, 'vocab_len num_states num_states']
    state_names_dict: Dict[str, int]
    vocab_len: int
    num_states: int
    steady_state: Float[np.ndarray, 'num_states']

    @property
    def steady_state(self):
        return calculate_steady_state_distribution(self.transition_matrix)

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

        # Set the number of symbols and states based on the shape of the transition matrix
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
        p = self.transition_matrix[:, current_state_index, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        return emission

    def generate_single_sequence(self, total_length: int, return_states: bool) -> ProcessHistory:
        """
        Generate a sequence of states based on the transition matrix.
        """        
        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        # randomly select state from steady state distribution
        current_state_ind = np.random.choice(self.num_states, p=self.steady_state)

        symbols = []
        states = []
        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_ind])
           
            emission = self._sample_emission(current_state_ind)
            symbols.append(emission)
           
            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(self.transition_matrix[emission, current_state_ind, :])
            current_state_ind = next_state_ind
        return ProcessHistory(symbols=symbols, states=states if return_states else None)
        
    def generate_multiple_sequences(self, num_sequences: int, total_length: int, return_states: bool) -> List[ProcessHistory]:
        return [self.generate_single_sequence(total_length, return_states) for _ in range(num_sequences)]