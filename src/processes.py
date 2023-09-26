import random
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

class RRXORProcess:
    """
    Class for generating Random-Random-XOR (RRXOR) data.
    """

    @staticmethod
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
        
        # Start the sequence randomly at bit 1,2 or 3
        start_index = random.randint(0, 2)
        output = output[start_index:]
        positions = positions[start_index:]

        # Return the sequence up to the desired total length along with positions
        if with_positions:
            return output[:total_length], positions[:total_length]
        else:
            return output[:total_length]
        
    @staticmethod
    def get_epsilon_machine(with_state_names=False):
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
        T[0, state_names['S'], state_names['0']] = 0.5
        T[1, state_names['S'], state_names['1']] = 0.5
        T[0, state_names['0'], state_names['F']] = 0.5
        T[1, state_names['0'], state_names['T']] = 0.5
        T[0, state_names['1'], state_names['T']] = 0.5
        T[1, state_names['1'], state_names['F']] = 0.5
        T[1, state_names['T'], state_names['S']] = 1.0
        T[0, state_names['F'], state_names['S']] = 1.0

        if with_state_names:
            return T, state_names
        else:
            return T
        
    @staticmethod
    def create_train_test_data(sequence, input_size, split_ratio=0.8):
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
    
    @staticmethod

    def create_data_loader(data_inputs, data_targets, batch_size=64):
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
        
    
        

