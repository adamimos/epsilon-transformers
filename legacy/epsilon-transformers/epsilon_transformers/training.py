import torch
import numpy as np
import random
import wandb
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torch.utils.data import DataLoader, Dataset
from torch import optim
from typing import Dict, Tuple
from epsilon_transformers.comp_mech import (
    generate_sequences,
    collect_path_probs_with_paths,
)
from epsilon_transformers.comp_mech import HMM, Mixed_State_Tree


def set_random_seed(seed):
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


class SequenceDataset(Dataset):
    def __init__(self, data: np.ndarray, n_ctx: int):
        self.data = data
        self.n_ctx = n_ctx

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.n_ctx)

    def __getitem__(self, idx):
        sequence_idx = idx // (self.data.shape[1] - self.n_ctx)
        token_idx = idx % (self.data.shape[1] - self.n_ctx)
        X = self.data[sequence_idx, token_idx : token_idx + self.n_ctx]
        Y = self.data[sequence_idx, token_idx + 1 : token_idx + self.n_ctx + 1]
        return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)


def create_train_loader(
    data: np.ndarray, n_ctx: int, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for training data.

    Parameters:
    data (np.ndarray): The training data. Of shape (num_sequences, sequence_length).
    n_ctx (int): The context length.
    batch_size (int): The size of each batch.
    shuffle (bool): Whether to shuffle the data.

    Returns:
    DataLoader: The DataLoader for training data.
    """
    dataset = SequenceDataset(data, n_ctx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def create_validation_set(
    MSP_tree: Mixed_State_Tree.Mixed_State_Tree,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a validation set for the given mixed state tree.

    Parameters:
    MSP_tree (Mixed_State_Tree): The mixed state tree to create a validation set for.
    sequence_length (int): The length of the sequences in the validation set.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing the validation sequences and their probabilities.
    which can be used as weights for the loss function.
    """

    path_probs = collect_path_probs_with_paths(MSP_tree, sequence_length + 1)
    seqs = np.array([path[0] for path in path_probs])  # *(num_paths, sequence_length)
    probs = np.array([path[1] for path in path_probs])  # *(num_paths)

    # the X and Ys are one token shifted
    X = seqs[:, :-1]
    Y = seqs[:, 1:]

    return X, Y, probs


def build_dataset(config: Dict, process: HMM.HMM) -> np.ndarray:
    data = generate_sequences(
        process,
        num_sequences=config["num_sequences"],
        sequence_length=config["sequence_length"],
    )

    data = create_train_loader(
        data, batch_size=config["batch_size"], n_ctx=config["n_ctx"]
    )

    return data


def build_probabilistic_dataset(
    true_probs: np.ndarray, batch_size: int, num_iters: int
) -> np.ndarray:
    # multinomial sampling
    train_weights = np.random.multinomial(
        batch_size, true_probs, size=num_iters
    )  # *(num_iters, num_paths)
    # normalize by batch size for each iteration
    # convert batch_size to float to avoid integer division
    batch_size_float = float(batch_size)
    train_probs = train_weights / batch_size_float  # *(num_iters, num_paths)

    return train_probs


def build_network(s_config: Dict, device: torch.device) -> HookedTransformer:
    config = HookedTransformerConfig(
        d_model=s_config["d_model"],
        d_head=s_config["d_head"],
        n_layers=s_config["n_layers"],
        n_ctx=s_config["n_ctx"],
        n_heads=s_config["n_heads"],
        d_mlp=4 * s_config["d_model"],
        d_vocab=s_config["d_vocab"],
        act_fn=s_config["act_fn"],
        use_attn_scale=s_config["use_attn_scale"],
        normalization_type=s_config["normalization_type"],
        attention_dir=s_config["attention_dir"],
        attn_only=s_config["attn_only"],
        seed=s_config["seed"],
        init_weights=s_config["init_weights"],
        device=device,
    )

    model = HookedTransformer(config)

    return model


def build_optimizer(
    network: torch.nn.Module, sweep_config: Dict
) -> torch.optim.Optimizer:
    optimizer_type = sweep_config["optimizer"]
    learning_rate = sweep_config["learning_rate"]
    weight_decay = sweep_config["weight_decay"]

    if optimizer_type == "adam":
        optimizer = optim.Adam(
            network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(
            network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    verbose=True,
):
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

            running_acc += compute_accuracy(
                torch.argmax(outputs, dim=-1)[:, -1], batch_targets[:, -1]
            )
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
                    batch_inputs, batch_targets = (
                        batch_inputs.cuda(),
                        batch_targets.cuda(),
                    )

                # Get model predictions
                outputs = model(batch_inputs)
                predicted_classes = torch.argmax(outputs, dim=-1)

                # Compute overall accuracy
                overall_accuracy = compute_accuracy(predicted_classes, batch_targets)
                overall_accuracies.append(overall_accuracy)

                # Compute accuracy for the last bit
                last_bit_accuracy = compute_accuracy(
                    predicted_classes[:, -1], batch_targets[:, -1]
                )
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


def train_hooked_model(
    model_config,
    train_loader,
    test_loader,
    criterion,
    num_epochs,
    verbose=True,
    device="cuda",
):
    # move everything to device,
    # in particular the data

    d_vocab = model_config.d_vocab
    # Initialize your model with the fixed architecture configuration
    model = HookedTransformer(model_config)

    # Retrieve hyperparameters for this run from wandb.config
    learning_rate = wandb.config.learning_rate
    weight_decay = wandb.config.weight_decay

    # Set up the optimizer with these hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

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

            running_acc += compute_accuracy(
                torch.argmax(outputs, dim=-1)[:, -1], batch_targets[:, -1]
            )
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
                    batch_inputs, batch_targets = (
                        batch_inputs.cuda(),
                        batch_targets.cuda(),
                    )

                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                # Get model predictions
                outputs = model(
                    batch_inputs
                )  # outputs is of size (batch_size, input_size, d_vocab)
                predicted_classes = torch.argmax(outputs, dim=-1)

                # Compute overall accuracy
                overall_accuracy = compute_accuracy(predicted_classes, batch_targets)
                overall_accuracies.append(overall_accuracy)

                # Compute accuracy for the last bit
                last_bit_accuracy = compute_accuracy(
                    predicted_classes[:, -1], batch_targets[:, -1]
                )
                last_bit_accuracies.append(last_bit_accuracy)

                # Compute loss for the last bit
                last_bit_loss = criterion(outputs[:, -1, :], batch_targets[:, -1])
                last_bit_losses.append(last_bit_loss.item())

            # Calculate average accuracies and losses for the entire test set after each epoch
            avg_overall_accuracy = sum(overall_accuracies) / len(overall_accuracies)
            avg_last_bit_accuracy = sum(last_bit_accuracies) / len(last_bit_accuracies)
            avg_last_bit_loss = sum(last_bit_losses) / len(
                last_bit_losses
            )  # Calculate average last bit loss

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_accuracy": avg_training_acc,
                "train_loss": loss.item(),
                "test_accuracy": avg_overall_accuracy,
                "test_last_bit_accuracy": avg_last_bit_accuracy,
                "test_last_bit_loss": avg_last_bit_loss,  # Log the average last bit loss
            }
        )

        # Print the results in a tabulated format
        if verbose:
            if epoch == 0:
                header = (
                    "| Epoch | Training Acc. | Loss | Overall Acc. | Last Bit Acc. |"
                )
                print(header)
            row = f"| {epoch+1:^5} | {avg_training_acc:^12.2%} | {loss.item():^13.4f} | {avg_overall_accuracy:^17.2%} | {avg_last_bit_accuracy:^16.2%} |"
            print(row)

    return model


def sweep_train(process, model_config, train_config):
    with wandb.init() as run:
        # Prepare your data loaders and criterion
        train_loader, test_loader, sequence_positions = process.prepare_data(
            train_config["sequence_length"],
            train_config["num_sequences"],
            model_config.n_ctx,
            split_ratio=0.8,
            batch_size=train_config["batch_size"],
            with_positions=True,
        )

        run_name = f"RRXOR_lr_{run.config.learning_rate}-wd_{run.config.weight_decay}"
        run.name = run_name
        run.config.update(
            {"model_config": model_config.__dict__, "train_config": train_config}
        )

        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 200  # or any other value you wish to fix

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Call the training function with the model config
        train_hooked_model(
            model_config,
            train_loader,
            test_loader,
            criterion,
            num_epochs,
            device=device,
        )
