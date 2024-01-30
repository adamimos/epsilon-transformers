from epsilon_transformers.processes import RRXOR
from epsilon_transformers.visualization import visualize_graph
from epsilon_transformers.simple_transformer import train_hooked_model
from epsilon_transformers.markov_utilities import calculate_sequence_probabilities, compute_myopic_entropy_from_MSP, to_mixed_state_presentation
import wandb
import yaml
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Load the config file
with open("./RRXOR_sweep_cfg.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if config['process'] == 'RRXOR':
    process = RRXOR()

# Sweep over the config file

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def sweep_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config


        X, Y, test_weights, epoch_weights = build_dataset(process, config.batch_size, 
                                                  config.num_epochs, 
                                                  config.n_ctx)
        
        # convert to torch tensors
        X = torch.tensor(X).to(device)
        Y = torch.tensor(Y).to(device)
        test_weights = torch.tensor(test_weights, dtype=torch.float32).to(device)
        epoch_weights = torch.tensor(epoch_weights, dtype=torch.float32).to(device)


        # compute optimal performance via myopic entropy rate!
        MSP = to_mixed_state_presentation(process.transition_matrix, threshold=1e-5)
        myopic_entropy = compute_myopic_entropy_from_MSP(MSP,config.n_ctx+1)
        first_position_optimal_loss = myopic_entropy[1]
        last_position_optimal_loss = myopic_entropy[-1]

        first_position_optimal_loss = torch.tensor(first_position_optimal_loss, dtype=torch.float32).to(device)
        last_position_optimal_loss = torch.tensor(last_position_optimal_loss, dtype=torch.float32).to(device)

        network = build_network(
            d_model=config['d_model'], 
            d_head=config['d_head'], 
            n_layers=config['n_layers'], 
            n_ctx=config['n_ctx'], 
            n_heads=config['n_heads'], 
            d_mlp=config['d_mlp'], 
            d_vocab=config['d_vocab'], 
            act_fn=config['act_fn'], 
            use_attn_scale=config['use_attn_scale'], 
            normalization_type=config['normalization_type'], 
            attention_dir=config['attention_dir'], 
            attn_only=config['attn_only'], 
            seed=config['seed'], 
            init_weights=config['init_weights'], 
            device=device
        )
        optimizer = build_optimizer(network, config['optimizer'], config['learning_rate'], config['weight_decay'])

        for epoch in range(config.num_epochs):

            train_loss_per_position, train_loss_mean, test_loss_per_position, test_loss_mean = train_epoch(network, optimizer,
                                                                                                            X, Y, epoch_weights[epoch],
                                                                                                            test_weights, config.d_vocab)
            
            test_loss_initial = test_loss_per_position[0].item()
            test_loss_end = test_loss_per_position[-1].item()
            percent_initial_loss = test_loss_initial / first_position_optimal_loss
            percent_end_loss = test_loss_end / last_position_optimal_loss


            wandb.log({'train_loss_per_position': train_loss_per_position,
                       'train_loss_mean': train_loss_mean,
                       'test_loss_per_position': test_loss_per_position,
                       'test_loss_mean': test_loss_mean,
                       'test_loss_initial': test_loss_initial,
                       'test_loss_end': test_loss_end,
                       'percent_initial_loss': percent_initial_loss,
                       'percent_end_loss': percent_end_loss,})
            
                        # check if nan and if so break
            if torch.isnan(train_loss_mean):
                break


"""
def build_dataset(process, batch_size, sequence_length, num_sequences, n_ctx):
    train_loader, test_loader, pos = process.prepare_data(
        sequence_length,
        num_sequences,
        n_ctx,
        split_ratio=0.8,
        batch_size=batch_size,
        with_positions=True
    )
    return train_loader, test_loader
"""
def build_dataset(process, batch_size, num_epochs, n_ctx):
    """
    Constructs the dataset for training by calculating the sequence probabilities,
    generating weights for each epoch, and preparing the input and target sequences.

    Args:
        process: The data processing object with a method to calculate sequence probabilities.
        batch_size: The number of samples per batch.
        num_epochs: The total number of epochs for which to generate weights.
        n_ctx: The context size (number of tokens) for each sequence.

    Returns:
        A tuple containing:
        - Input sequences for the model (X)
        - Target sequences for the model (Y)
        - Array of ground truth probabilities for each sequence
        - Weights for each epoch, normalized by batch size
    """
    # Calculate sequence probabilities for sequences of length n_ctx+1
    sequence_probabilities = calculate_sequence_probabilities(process.transition_matrix, n_ctx+1)
    sequence_probabilities = sequence_probabilities[n_ctx+1]
    ground_truth_probabilities = list(sequence_probabilities.values())
    sequences = list(sequence_probabilities.keys())

    # Initialize random number generator
    rng = np.random.default_rng()
    # Generate weights for each epoch, normalized by batch size
    epoch_weights = rng.multinomial(batch_size, ground_truth_probabilities, size=num_epochs) / batch_size
    # Convert sequences to numpy arrays for processing
    sequences_array = np.array([np.array([int(token) for token in sequence]) for sequence in sequences])
    # Prepare input (X) and target (Y) sequences
    input_sequences = sequences_array[:, :-1]  # Exclude the last token for inputs
    target_sequences = sequences_array[:, 1:]  # Exclude the first token for targets

    return input_sequences, target_sequences, np.array(ground_truth_probabilities), epoch_weights




def build_network(d_model, d_head, n_layers, n_ctx, n_heads, d_mlp, d_vocab,
                  act_fn, use_attn_scale, normalization_type, attention_dir,
                  attn_only, seed, init_weights, device):
    
    config = HookedTransformerConfig(
        d_model=d_model,
        d_head=d_head,
        n_layers=n_layers,
        n_ctx=n_ctx,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        act_fn=act_fn,
        use_attn_scale=use_attn_scale,
        normalization_type=normalization_type,
        attention_dir=attention_dir,
        attn_only=attn_only,
        seed=seed,
        init_weights=init_weights,
        device=device,
    )

    model = HookedTransformer(config)

    return model

def build_optimizer(network, optimizer_type, weight_decay, learning_rate):

    if optimizer_type == 'adam':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer

def train_epoch(model, optimizer, X, Y, train_weights , test_weights, d_vocab):
    model.train()
    running_acc = 0.0
    num_batches = 0
    epoch_train_loss = 0.0

    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer.zero_grad()
    
    # training run
    outputs = model(X) # outputs is of size (batch_size, input_size, d_vocab)
    loss = criterion(outputs.view(-1, d_vocab), Y.view(-1)) # (batch_size * input_size)
    # train weights is shape (batch_size), so for each batch we need to repeat it input_size times
    train_weights = train_weights.repeat(X.shape[1])
    train_loss = loss * train_weights.view(-1) # (batch_size * input_size)
    train_loss = train_loss.view(X.shape[0], X.shape[1]) # (batch_size, input_size)
    train_loss_per_position = train_loss.sum(dim=0) # (input_size)
    train_loss_mean = train_loss_per_position.mean()
    train_loss_mean.backward()
    optimizer.step()

    # testing run (its really for epoch-1 but whatever, close enough!)
    test_weights = test_weights.repeat(X.shape[1])
    test_loss = loss * test_weights.view(-1) # (batch_size * input_size)
    test_loss = test_loss.view(X.shape[0], X.shape[1]) # (batch_size, input_size)
    test_loss_per_position = test_loss.sum(dim=0) # (input_size)
    test_loss_mean = test_loss_per_position.mean()

    # calculating training accuracy
    """    
    predicted_classes = torch.argmax(outputs, dim=-1) # (batch_size, input_size)
    correct_preds = (predicted_classes == Y).float() # (batch_size, input_size)
    train_acc = correct_preds * train_weights # (batch_size * input_size)
    train_acc_mean = train_acc.mean()
    test_acc = correct_preds * test_weights # (batch_size * input_size)
    test_acc_mean = test_acc.mean()
    """

    # return all the losses and things we want to log
    return train_loss_per_position, train_loss_mean, test_loss_per_position, test_loss_mean
    



"""
def train_epoch(model, train_loader, test_loader, optimizer, d_vocab):
    model.train()
    running_acc = 0.0
    num_batches = 0
    epoch_train_loss = 0.0

    criterion = nn.CrossEntropyLoss()
    

    for batch_inputs, batch_targets in train_loader:
        if torch.cuda.is_available():
            batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs.view(-1, d_vocab), batch_targets.view(-1))
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        running_acc += compute_accuracy(torch.argmax(outputs, dim=-1)[:,-1], batch_targets[:,-1])
        num_batches += 1

    # Calculate training results after each epoch
    avg_training_acc = running_acc / num_batches
    avg_training_loss = epoch_train_loss / num_batches
    
    # Evaluation on test set after each epoch
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # no gradients needed for evaluation
        overall_accuracies = []
        last_bit_accuracies = []
        last_bit_losses = []  # Add a list to store the last bit losses

        # Iterate over test batches
        for idx, (batch_inputs, batch_targets) in enumerate(test_loader):
            if torch.cuda.is_available():
                batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()
            
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Get model predictions
            outputs = model(batch_inputs) # outputs is of size (batch_size, input_size, d_vocab)
            predicted_classes = torch.argmax(outputs, dim=-1)

            # Compute overall accuracy
            overall_accuracy = compute_accuracy(predicted_classes, batch_targets)
            overall_accuracies.append(overall_accuracy)

            # Compute accuracy for the last bit
            last_bit_accuracy = compute_accuracy(predicted_classes[:, -1], batch_targets[:, -1])
            last_bit_accuracies.append(last_bit_accuracy)

            # Compute loss for the last bit
            last_bit_loss = criterion(outputs[:, -1, :], batch_targets[:, -1])
            last_bit_losses.append(last_bit_loss.item())

        # Calculate average accuracies and losses for the entire test set after each epoch
        avg_overall_accuracy = sum(overall_accuracies) / len(overall_accuracies)
        avg_last_bit_accuracy = sum(last_bit_accuracies) / len(last_bit_accuracies)
        avg_last_bit_loss = sum(last_bit_losses) / len(last_bit_losses)  # Calculate average last bit loss
    
    # return all the losses and things we want to log
    return avg_training_loss, avg_overall_accuracy, avg_last_bit_accuracy, avg_last_bit_loss
    """

def compute_accuracy(predictions, targets):
    """
    Compute accuracy for predictions against targets.
    """
    correct_preds = (predictions == targets).float()
    accuracy = correct_preds.mean().item()
    return accuracy


sweep_id = wandb.sweep(config, project=config['sweep_name'])
wandb.agent(sweep_id, function=sweep_train, count=100)

