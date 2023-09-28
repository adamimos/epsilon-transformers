import random
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from markov_utilities import calculate_steady_state_distribution


class Process:
    """
    Parent class for generating process data.
    """

    def __init__(self):
        self.T, self.state_names = self._get_epsilon_machine(with_state_names=True)
        self.steady_state = calculate_steady_state_distribution(self.T)

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        raise NotImplementedError("This method should be overridden by child class")

    def generate(self, total_length, with_positions=False):
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
        num_states = self.T.shape[1]
        current_state = np.random.choice(num_states, p=self.steady_state)

        sequence = []
        positions = []
        for _ in range(total_length):
            if with_positions:
                positions.append(self.state_names[current_state])
            # randomly select output based on transition matrix
            p = self.T[:, current_state, :].sum(axis=1)
            emission = np.random.choice(2, p=p)
            # make transition
            next_state = np.random.choice(num_states, p=self.T[emission, current_state, :])
            sequence.append(next_state)
            current_state = next_state

        if with_positions:
            return sequence, positions
        else:
            return sequence
    
    def prepare_data(self, total_length, input_size, split_ratio=0.8, batch_size=64, with_positions=False):
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
            sequence, positions = self.generate(total_length, with_positions)
        else:
            sequence = self.generate(total_length, with_positions)

        # Create training and testing data
        train_inputs, train_targets, test_inputs, test_targets = self.create_train_test_data(sequence, input_size, split_ratio)

        # Create data loaders
        train_loader = self.create_data_loader(train_inputs, train_targets, batch_size)
        test_loader = self.create_data_loader(test_inputs, test_targets, batch_size)

        if with_positions:
            return train_loader, test_loader, positions
        else:
            return train_loader, test_loader
    
    def create_train_test_data(self, sequence, input_size, split_ratio=0.8):
        """
        Create training and testing data from a sequence.

        Parameters:
        sequence (str): The input sequence.
        input_size (int): The size of the input to be used for prediction.
        split_ratio (float): The ratio of data to be used for training. Default is 0.8.

        Returns:
        list: Training inputs.
        list: Training targets.
        list: Testing inputs.
        list: Testing targets.
        """
        inputs, targets = [], []
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

class RRXORProcess(Process):

    def __init__(self, pR1=0.5, pR2=0.5):
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the RRXOR process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
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

    def generate(total_length, with_positions=False):
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

class GoldenMeanProcess(Process):
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
        
    
class ZeroOneRProcess(Process):
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