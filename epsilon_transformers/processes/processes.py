import random
import numpy as np
import torch
import torch.utils.data
from epsilon_transformers.analysis.markov_utilities import calculate_steady_state_distribution
import networkx as nx
from typing import List, Tuple

NUM_STATES = 5
NUM_SYMBOLS = 2
ALPHA = 1.0


class Presentation:
    """
    This is the parent class for presentations of a process. It is responsible for initializing the transition matrix,
    calculating the steady state distribution, and setting the number of symbols and states.
    """

    def __init__(self, transition_matrix: np.ndarray = None, state_names: dict = None):
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
            self.state_names = state_names if state_names is not None else {str(i): i for i in range(self.transition_matrix.shape[1])}
            # have another property which is the reverse of state_names
            self.state_inds = {v: k for k, v in self.state_names.items()}
        else:
            self.transition_matrix, self.state_names = self._get_epsilon_machine(with_state_names=True)
            self.state_inds = {v: k for k, v in self.state_names.items()}

        # Calculate the steady state distribution of the transition matrix
        self.steady_state = calculate_steady_state_distribution(self.transition_matrix)
        
        # Set the number of symbols and states based on the shape of the transition matrix
        self.num_symbols = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

        # Check that transition matrix has 3 axes and that the first and second dim are the same size
        if len(self.transition_matrix.shape) != 3 or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should have 3 axes and the final two dims shoulds be square")

    def is_unifilar(self):
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

        

    @staticmethod
    def check_if_valid(self):
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
    
    
    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the presentation.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        raise NotImplementedError("This method should be overridden by child class")

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

class RRXOR(Presentation):

    def __init__(self, pR1=0.5, pR2=0.5):
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the RRXOR process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping
                                state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if
              with_state_names is True.
        """
        T = np.zeros((2, 5, 5))
        state_names = {'S': 0, '0': 1, '1': 2, 'T': 3, 'F': 4}
        T[0, state_names['S'], state_names['0']] = self.pR1
        T[1, state_names['S'], state_names['1']] = 1 - self.pR1
        T[0, state_names['0'], state_names['F']] = self.pR2
        T[1, state_names['0'], state_names['T']] = 1 - self.pR2
        T[0, state_names['1'], state_names['T']] = self.pR2
        T[1, state_names['1'], state_names['F']] = 1 - self.pR2
        T[1, state_names['T'], state_names['S']] = 1.0
        T[0, state_names['F'], state_names['S']] = 1.0

        if with_state_names:
            return T, state_names
        else:
            return T

    def generate_without_epsilon_machine(total_length, with_positions=False):
        """
        Generate a sequence of Random-Random-XOR (RRXOR) data.

        Parameters:
        total_length (int): The total length of the sequence to generate.
        with_positions (bool): If True, also return a list of positions ("R1", "R2", "XOR").

        Returns:
        list: The generated RRXOR sequence. If with_positions is True, also return a list of positions.
        """
        output = []
        positions = []
        
        while len(output) < total_length+3:
            bit1 = random.randint(0, 1)
            bit2 = random.randint(0, 1)
            xor_result = bit1 ^ bit2
            output.extend([str(bit1), str(bit2), str(xor_result)])
            positions.extend(["R1", "R2", "XOR"])
        
        # Start the sequence randomly at bit 1,2 r 3
        start_index = random.randint(0, 2)
        output = output[start_index:]
        positions = positions[start_index:]

        # Return the sequence up to the desired total length along with positions
        if with_positions:
            return output[:total_length], positions[:total_length]
        else:
            return output[:total_length]

class GoldenMean(Presentation):
    """
    Class for generating RKGoldenMean data.
    """
    def __init__(self, R, k, p):
        """
        Initialize the GoldenMeanProcess with R, k, p parameters.

        Parameters:
        R (int): The number of states that output 1.
        k (int): The number of states that output 0.
        p (float): The probability of outputting 1 in the final state.
        """
        self.R = R
        self.k = k
        self.p = p

        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the RKGoldenMean process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        assert self.k <= self.R, "k should be less than or equal to R"

        n_states = self.R + self.k
        T = np.zeros((2, n_states, n_states))

        # State names
        state_names = {chr(65 + i): i for i in range(n_states)}  # chr(65) is 'A'

        # First state
        T[1, state_names['A'], state_names['B']] = self.p
        T[0, state_names['A'], state_names['A']] = 1 - self.p

        # States that output 1
        for i in range(1, self.R):
            T[1, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # States that output 0
        for i in range(self.R, self.R+self.k-1):
            T[0, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # Last state
        T[0, state_names[chr(65 + n_states - 1)], state_names['A']] = 1.0

        if with_state_names:
            return T, state_names
        else:
            return T
        
    
class ZeroOneR(Presentation):
    """
    Class for generating 01R data.
    """

    def __init__(self, p=0.5):
        self.p = p # probability of emitting 0 from the R state
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the 01R process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 3, 3))
        state_names = {'0': 0, '1': 1, 'R': 2}
        T[0, state_names['0'], state_names['1']] = 1.0
        T[1, state_names['1'], state_names['R']] = 1.0
        T[0, state_names['R'], state_names['0']] = self.p
        T[1, state_names['R'], state_names['0']] = 1-self.p

        if with_state_names:
            return T, state_names
        else:
            return T
        

class Even(Presentation):
    """
    Class for generating EvenProcess data.
    """

    def __init__(self, p=2/3):
        self.p = p
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the EvenProcess.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 2, 2))
        state_names = {'E': 0, 'O': 1}
        T[1, state_names['E'], state_names['O']] = 1-self.p
        T[0, state_names['E'], state_names['E']] = self.p
        T[1, state_names['O'], state_names['E']] = 1.0


        if with_state_names:
            return T, state_names
        else:
            return T
        


class Nond(Presentation):
    """
    Class for generating the nond process, as defined in
    """

    def __init__(self):
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the nond process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 3, 3))
        state_names = {'0': 0, '1': 1, '2': 2}
        T[0, 2, 0] = 1.0
        T[1, 0, 1] = 0.5
        T[1, 1, 1] = 0.5
        T[1, :, 2] = 1./3.

        if with_state_names:
            return T, state_names
        else:
            return T
        
class Mess3(Presentation):
    """
    Class for generating the Mess3 process, as defined in
    """

    def __init__(self, x=0.15, a=0.6):
        self.x = x
        self.a = a
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the Mess3 process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((3, 3, 3))
        state_names = {'A': 0, 'B': 1, 'C': 2}
        b = (1-self.a)/2
        y = 1-2*self.x

        ay = self.a*y
        bx = b*self.x
        by = b*y
        ax = self.a*self.x

        T[0, :, :] = [[ay, bx, bx],
                      [ax, by, bx],
                      [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx],
                      [bx, ay, bx],
                      [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax],
                      [bx, by, ax],
                      [bx, bx, ay]]


        if with_state_names:
            return T, state_names
        else:
            return T
