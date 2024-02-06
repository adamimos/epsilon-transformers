import numpy as np
from typing import Tuple, Optional, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
from jaxtyping import Float

from epsilon_transformers.markov_utilities import calculate_steady_state_distribution

# TODO: Turn check_if_valid into validator and use it to validate
# TODO: Use jaxtyping
# TODO: Decouple the visualization (networkx) from the 
# TODO: change _create_hmm so that it doesn't create an output. And change it's name as well

@dataclass
class ProcessHistory:
    symbols: List
    states: Optional[List] = None

class Process(ABC):
    """
    This is the parent class for presentations of a process. It is responsible for initializing the transition matrix,
    calculating the steady state distribution, and setting the number of symbols and states.
    """
    transition_matrix: Float[np.ndarray, 'vocab_len num_states num_states']
    state_names: Dict[str, int]
    vocab_len: int
    num_states: int
    steady_state: np.ndarray

    def __init__(self):
        """
        Initialize the Presentation object.

        Parameters:
        transition_matrix (numpy.ndarray, optional): The transition matrix to be used. If not provided, the epsilon machine
        will be used to generate the transition matrix. The matrix should have 3 axes: symbols, from state, to state.
        Each entry T(x)_ij represents Pr(s_j, x|s_i), the probability of transitioning from state s_i to state s_j given
        symbol x.
        state_names (dict, optional): A dictionary mapping state names (strings) to indices (ints). If not provided, 
        state names will be generated based on the shape of the transition matrix or the epsilon machine.
        """
        self.transition_matrix, self.state_names = self._create_hmm()

        # Check that transition matrix has 3 axes and that the first and second dim are the same size
        if len(self.transition_matrix.shape) != 3 or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should have 3 axes and the final two dims shoulds be square")

        # Check if the transition matrix is square
        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")
        
        # Check if the transition matrix is stochastic and sum to 1
        # if we sum over the first axis then we get transitions from state i to state j
        # now for every state if its unifilar we should have a sum of 1 when we sum over the second axis
        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        # Calculate the steady state distribution of the transition matrix
        self.steady_state = calculate_steady_state_distribution(self.transition_matrix)
        
        # Set the number of symbols and states based on the shape of the transition matrix
        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

    @abstractmethod
    def _create_hmm(self) -> Tuple[Float[np.ndarray, 'vocab_len num_states num_states'], Dict[str,int]]:
        """
        Generate the epsilon machine for the presentation.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        ...

    def is_unifilar(self) -> bool:
        """
        A presentation is unifilar if for every state and symbol, there is at most one transition to another state.
        """
        # For each state, check if there are multiple transitions for each symbol
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                # If there are multiple transitions, return False
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True


    def generate_single_sequence(self, total_length: int, return_states: bool) -> ProcessHistory:
        """
        Generate a sequence of states based on the transition matrix.

        Parameters:
        total_length (int): The total length of the sequence to generate.
        with_positions (bool): If True, also return a list of state names.

        Returns:
        list: The generated sequence of states.
        list: The state names. Only returned if with_positions is True.
        """
        # randomly select state from steady state distribution
        num_states = self.num_states
        num_symbols = self.vocab_len
        transition_matrix = self.transition_matrix
        
        # flip keys and vals for state names
        state_names = {v: k for k, v in self.state_names.items()}
        current_state_ind = np.random.choice(num_states, p=self.steady_state)

        symbols = []
        states = []
        for _ in range(total_length):
            states.append(state_names[current_state_ind])
           
            # randomly select output based on transition matrix
            p = self.transition_matrix[:, current_state_ind, :].sum(axis=1)
            emission = np.random.choice(num_symbols, p=p)
           
            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(transition_matrix[emission, current_state_ind, :])
            symbols.append(emission)
           
            current_state_ind = next_state_ind

        return ProcessHistory(symbols=symbols, states=states if return_states else None)
        
    def generate_multiple_sequences(self, total_length: int, num_sequences: int, with_positions=False) -> List[ProcessHistory]:
        """
        Generate multiple sequences of states based on the transition matrix.

        Parameters:
        total_length (int): The total length of each sequence to generate.
        num_sequences (int): The number of sequences to generate.
        with_positions (bool): If True, also return lists of state names.

        Returns:
        list: A list containing multiple sequences.
        list: A list of lists containing state names for each sequence. Only returned if with_positions is True.
        """
        return [self.generate_single_sequence(total_length, with_positions) for _ in range(num_sequences)]