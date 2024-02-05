import torch
import torch.utils.data
from typing import List

from epsilon_transformers.processes.process import Process


# Dataset Creation
def prepare_data(process: Process, total_length, num_sequences, input_size, split_ratio=0.8, batch_size=64, with_positions=False):
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
        sequence, positions = process.generate_multiple_sequences(total_length,num_sequences, with_positions)
    else:
        sequence = process.generate_multiple_sequences(total_length, num_sequences, with_positions)

    # Create training and testing data
    train_inputs, train_targets, test_inputs, test_targets = create_train_test_data(sequence, input_size, split_ratio)

    # Create data loaders
    train_loader = create_data_loader(train_inputs, train_targets, batch_size)
    test_loader = create_data_loader(test_inputs, test_targets, batch_size)

    if with_positions:
        return train_loader, test_loader, positions
    else:
        return train_loader, test_loader

def create_train_test_data(sequences, input_size, split_ratio=0.8):
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

def create_data_loader(data_inputs: List, data_targets: List, batch_size=64):
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