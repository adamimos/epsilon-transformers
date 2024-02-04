import numpy as np
import torch
import torch.utils.data
import networkx as nx
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod

from epsilon_transformers.markov_utilities import calculate_steady_state_distribution

NUM_STATES = 5
NUM_SYMBOLS = 2
ALPHA = 1.0

# TODO: Turn an epsilon machine into a class of it's own ??
# TODO: Turn check_if_valid into validator and use it to validate
# TODO: Use jaxtyping
# TODO: Decouple the visualization (networkx) from the 

class Presentation(ABC):
    """
    This is the parent class for presentations of a process. It is responsible for initializing the transition matrix,
    calculating the steady state distribution, and setting the number of symbols and states.
    """
    transition_matrix: np.ndarray
    state_names: Dict[str, int]
    num_symbols: int
    num_states: int
    steady_state: np.ndarray

    def __init__(self, transition_matrix: Optional[np.ndarray] = None, state_names: Optional[Dict[str, int]] = None):
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
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
            if state_names is not None:
                self.state_names = state_names  
            else:
                self.state_names = {str(i): i for i in range(self.transition_matrix.shape[1])} # ???
        else:
            self.transition_matrix, self.state_names = self._get_epsilon_machine(with_state_names=True)

        # Calculate the steady state distribution of the transition matrix
        self.steady_state = calculate_steady_state_distribution(self.transition_matrix)
        
        # Set the number of symbols and states based on the shape of the transition matrix
        self.num_symbols = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

        # Check that transition matrix has 3 axes and that the first and second dim are the same size
        if len(self.transition_matrix.shape) != 3 or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should have 3 axes and the final two dims shoulds be square")

    def is_unifilar(self) -> bool:
        """
        A presentation is unifilar if for every state and symbol, there is at most one transition to another state.
        """
        # For each state, check if there are multiple transitions for each symbol
        for i in range(self.num_states):
            for j in range(self.num_symbols):
                # If there are multiple transitions, return False
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    def check_if_valid(self) -> bool:
        """
        Check if the transition matrix is valid.
        """

        # Check if the transition matrix is square
        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")
        
        # Check if the transition matrix is stochastic and sum to 1
        # if we sum over the first axis then we get transitions from state i to state j
        # now for every state if its unifilar we should have a sum of 1 when we sum over the second axis
        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")
        return True

    @staticmethod
    def random_markov_chain(num_states: int, num_symbols: int, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Create and return a Markov chain as transition matrix and emission probabilities."""
        # Transition matrix, T[i,j]=k means that when we go from state i and emit j, we go to state k
        transition_matrix = np.random.randint(num_states, size=(num_states, num_symbols))
        
        # Emission probabilities using a Dirichlet distribution
        # this creates a matrix of size (num_states, num_symbols) where each row sums to 1
        emission_probabilities = np.random.dirichlet([alpha] * num_symbols, size=num_states)

        return transition_matrix, emission_probabilities

    @staticmethod
    def get_recurrent_subgraph(G: nx.DiGraph) -> nx.DiGraph:
        """Extract and return the largest strongly connected component from graph G."""
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        H = G.subgraph(largest_scc).copy()
        return H

    @staticmethod
    def transition_to_graph(transition_matrix: np.ndarray, num_symbols: int) -> nx.DiGraph:
        """Convert a transition matrix to a graph."""
        G = nx.DiGraph()
        for i in range(transition_matrix.shape[0]):
            for j in range(num_symbols):
                G.add_edge(i, transition_matrix[i, j], label=str(j))
        return G
        
    @staticmethod
    def recurrent_state_transition_matrices(state_transition_matrix: np.ndarray, 
                                            emission_probabilities: np.ndarray, 
                                            recurrent_nodes: List[int]) -> np.ndarray:
        """Construct transition matrices for recurrent states of a subgraph."""
        num_states = len(recurrent_nodes)
        num_symbols = emission_probabilities.shape[1]
        
        # Mapping of original state indices to recurrent indices
        state_mapping = {original: idx for idx, original in enumerate(recurrent_nodes)}
        
        # Create empty matrices for state transitions
        state_trans_matrices = [np.zeros((num_states, num_states)) for _ in range(2)]
        
        # Populate the matrices
        for original_idx in recurrent_nodes:
            for j in range(num_symbols):  # usually 2 symbols: 0 and 1
                next_state = state_transition_matrix[original_idx, j]
                if next_state in recurrent_nodes:
                    i = state_mapping[original_idx]
                    k = state_mapping[next_state]
                    state_trans_matrices[j][i, k] = emission_probabilities[original_idx, j]

        return np.array(state_trans_matrices)

    @classmethod
    def random(cls, num_states: int = NUM_STATES, num_symbols: int = NUM_SYMBOLS, alpha: float = ALPHA) -> 'Presentation':
        """Generate a random epsilon machine and return its recurrent subgraph and state transition matrices."""
        state_transition_matrix, emission_probabilities = cls.random_markov_chain(num_states, num_symbols, alpha)
        G = cls.transition_to_graph(state_transition_matrix, num_symbols)
        H = cls.get_recurrent_subgraph(G)
        
        # Extract state transition matrices for the recurrent subgraph
        recurrent_trans_matrices = cls.recurrent_state_transition_matrices(state_transition_matrix, emission_probabilities, H.nodes)

        return cls(recurrent_trans_matrices)
    
    @abstractmethod
    def _get_epsilon_machine(self, with_state_names=False) -> Tuple[np.ndarray, Dict[str,int]]:
        """
        Generate the epsilon machine for the presentation.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        ...

    def generate_single_sequence(self, total_length, with_positions=False):
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
        num_symbols = self.num_symbols
        transition_matrix = self.transition_matrix
        # flip keys and vals for state names
        state_names = {v: k for k, v in self.state_names.items()}
        current_state_ind = np.random.choice(num_states, p=self.steady_state)

        sequence = []
        positions = []
        for _ in range(total_length):
            if with_positions:
                positions.append(state_names[current_state_ind])
            # randomly select output based on transition matrix
            p = self.transition_matrix[:, current_state_ind, :].sum(axis=1)
            emission = np.random.choice(num_symbols, p=p)
            # make transition. given the current state and the emission, the next state is determined
            next_state_ind = np.argmax(transition_matrix[emission, current_state_ind, :])
            sequence.append(emission)
            current_state_ind = next_state_ind

        if with_positions:
            return sequence, positions
        else:
            return sequence
        

    def generate_data(self, total_length, num_sequences=1, with_positions=False):
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
        sequences = []
        positions = [] if with_positions else None

        for _ in range(num_sequences):
            seq, pos = self.generate_single_sequence(total_length, with_positions)
            sequences.append(seq)
            if with_positions:
                positions.append(pos)

        if with_positions:
            return sequences, positions
        else:
            return sequences
    
    def prepare_data(self, total_length, num_sequences, input_size, split_ratio=0.8, batch_size=64, with_positions=False):
        """
        Generate a sequence, create training and testing data, and create data loaders.

        Parameters:
        total_length (int): The total amount of data to generate.
        input_size (int): The size of the context window
        split_ratio (float): The ratio of data to be used for training. Default is 0.8.
        batch_size (int): The batch size for the DataLoader. Default is 64.
        with_positions (bool): If True, also return a list of positions ("R1", "R2", "XOR").

        Returns:
        DataLoader: A DataLoader object containing the training input and target data.
        DataLoader: A DataLoader object containing the testing input and target data.
        """
        # Generate a sequence
        if with_positions:
            sequence, positions = self.generate_data(total_length,num_sequences, with_positions)
        else:
            sequence = self.generate_data(total_length, num_sequences, with_positions)

        # Create training and testing data
        train_inputs, train_targets, test_inputs, test_targets = self.create_train_test_data(sequence, input_size, split_ratio)

        # Create data loaders
        train_loader = self.create_data_loader(train_inputs, train_targets, batch_size)
        test_loader = self.create_data_loader(test_inputs, test_targets, batch_size)

        if with_positions:
            return train_loader, test_loader, positions
        else:
            return train_loader, test_loader
        
    def prepare_data_weighted(self, L, n_epochs):
        """
        create a dataset with a weighted distribution of sequences
        this should be more efficient for cases where there are not many permitted sequences
        """

        # compute all length L sequences and their associated probabilities

        # for a given epoch we want to simulate batches for SGD
        distributions = np.random.multinomial(n_epochs, probs)

    def create_train_test_data(self, sequences, input_size, split_ratio=0.8):
        """
        Create training and testing data from a list of sequences.

        Parameters:
        sequences (list): The list of input sequences.
        input_size (int): The size of the input to be used for prediction.
        split_ratio (float): The ratio of data to be used for training. Default is 0.8.

        Returns:
        list: Training inputs.
        list: Training targets.
        list: Testing inputs.
        list: Testing targets.
        """
        inputs, targets = [], []
        for sequence in sequences:
            for i in range(len(sequence) - input_size):
                input_seq = sequence[i:i+input_size]
                target_seq = sequence[i+1:i+input_size+1]  # Shifted by one position for next bit prediction
                inputs.append([int(bit) for bit in input_seq])
                targets.append([int(bit) for bit in target_seq])

        # Split into Training and Test Data
        split_idx = int(split_ratio * len(inputs))
        train_inputs, train_targets = inputs[:split_idx], targets[:split_idx]
        test_inputs, test_targets = inputs[split_idx:], targets[split_idx:]

        return train_inputs, train_targets, test_inputs, test_targets

    def create_data_loader(self, data_inputs, data_targets, batch_size=64):
        """
        Create a DataLoader from input and target data.

        Parameters:
        data_inputs (list): The input data.
        data_targets (list): The target data.
        batch_size (int): The batch size for the DataLoader. Default is 64.

        Returns:
        DataLoader: A DataLoader object containing the input and target data.
        """
        data_inputs, data_targets = torch.tensor(data_inputs, dtype=torch.long), torch.tensor(data_targets, dtype=torch.long)
        data = torch.utils.data.TensorDataset(data_inputs, data_targets)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        return data_loader