import torch
from transformer_lens import HookedTransformer
import random
import numpy as np
import wandb

# TODO: Include the typing information
# TODO: Generalize the training loop so that it's 

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(predictions, targets):
    """
    Compute accuracy for predictions against targets.
    """
    correct_preds = (predictions == targets).float()
    accuracy = correct_preds.mean().item()
    return accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, verbose=True):
    d_vocab = model.unembedding.out_features
    for epoch in range(num_epochs):
        model.train()
        running_acc = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            if torch.cuda.is_available():
                batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.view(-1, d_vocab), batch_targets.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            running_acc += compute_accuracy(torch.argmax(outputs, dim=-1)[:,-1], batch_targets[:,-1])
            num_batches += 1

        # Calculate training results after each epoch
        avg_training_acc = running_acc / num_batches
        
        # Evaluation on test set after each epoch
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():  # no gradients needed for evaluation
            overall_accuracies = []
            last_bit_accuracies = []
            last_bit_losses = []

            # Iterate over test batches
            for idx, (batch_inputs, batch_targets) in enumerate(test_loader):
                if torch.cuda.is_available():
                    batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

                # Get model predictions
                outputs = model(batch_inputs)
                predicted_classes = torch.argmax(outputs, dim=-1)

                # Compute overall accuracy
                overall_accuracy = compute_accuracy(predicted_classes, batch_targets)
                overall_accuracies.append(overall_accuracy)

                # Compute accuracy for the last bit
                last_bit_accuracy = compute_accuracy(predicted_classes[:, -1], batch_targets[:, -1])
                last_bit_accuracies.append(last_bit_accuracy)

                # Compute cross entropy loss for the last bit
                last_bit_loss = criterion(outputs[:, -1, :], batch_targets[:, -1])
                last_bit_losses.append(last_bit_loss.item())

            # Calculate average accuracies and loss for the entire test set after each epoch
            avg_overall_accuracy = sum(overall_accuracies) / len(overall_accuracies)
            avg_last_bit_accuracy = sum(last_bit_accuracies) / len(last_bit_accuracies)
            avg_last_bit_loss = sum(last_bit_losses) / len(last_bit_losses)
        
        # Print the results in a tabulated format
        if verbose:
            if epoch == 0:
                header = "| Epoch | Training Accuracy | Loss | Overall Accuracy | Last Bit Accuracy | Last Bit Loss |"
                print(header)

            # convert all losses to base2
            loss = loss.item() / np.log(2)
            avg_last_bit_loss = avg_last_bit_loss / np.log(2)
            avg_training_acc = avg_training_acc / np.log(2)
            avg_overall_accuracy = avg_overall_accuracy / np.log(2)
            avg_last_bit_accuracy = avg_last_bit_accuracy / np.log(2)

            row = f"| {epoch+1:^5} | {avg_training_acc:^17.2%} | {loss.item():^4.4f} | {avg_overall_accuracy:^16.2%} | {avg_last_bit_accuracy:^16.2%} | {avg_last_bit_loss:^4.4f} |"
            print(row)

    return model


def train_hooked_model(model_config, train_loader, test_loader, criterion, num_epochs, verbose=True, device='cuda'):

    # move everything to device,
    # in particular the data




    d_vocab = model_config.d_vocab
    # Initialize your model with the fixed architecture configuration
    model = HookedTransformer(model_config)

    # Retrieve hyperparameters for this run from wandb.config
    learning_rate = wandb.config.learning_rate
    weight_decay = wandb.config.weight_decay

    # Set up the optimizer with these hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        running_acc = 0.0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            if torch.cuda.is_available():
                batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs.view(-1, d_vocab), batch_targets.view(-1))
            loss.backward()
            optimizer.step()

            running_acc += compute_accuracy(torch.argmax(outputs, dim=-1)[:,-1], batch_targets[:,-1])
            num_batches += 1

        # Calculate training results after each epoch
        avg_training_acc = running_acc / num_batches
        
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
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_accuracy': avg_training_acc,
            'train_loss': loss.item(),
            'test_accuracy': avg_overall_accuracy,
            'test_last_bit_accuracy': avg_last_bit_accuracy,
            'test_last_bit_loss': avg_last_bit_loss  # Log the average last bit loss
        })

        # Print the results in a tabulated format
        if verbose:
            if epoch == 0:
                header = "| Epoch | Training Acc. | Loss | Overall Acc. | Last Bit Acc. |"
                print(header)
            row = f"| {epoch+1:^5} | {avg_training_acc:^12.2%} | {loss.item():^13.4f} | {avg_overall_accuracy:^17.2%} | {avg_last_bit_accuracy:^16.2%} |"
            print(row)


    return model

def sweep_train(process, model_config, train_config):
    with wandb.init() as run:
        # Prepare your data loaders and criterion
        train_loader, test_loader, sequence_positions = process.prepare_data(
            train_config['sequence_length'],
            train_config['num_sequences'],
            model_config.n_ctx,
            split_ratio=0.8,
            batch_size=train_config['batch_size'],
            with_positions=True
        )

        run_name = f"RRXOR_lr_{run.config.learning_rate}-wd_{run.config.weight_decay}"
        run.name = run_name
        run.config.update({
            'model_config': model_config.__dict__,
            'train_config': train_config
        })

        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 200  # or any other value you wish to fix

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Call the training function with the model config
        train_hooked_model(model_config, train_loader, test_loader, criterion, num_epochs, device=device)