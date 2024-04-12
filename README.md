# Epsilon Transformers

This project contains Python code for generating process data and training transformer models on it. The codebase is organized into several Python scripts and Jupyter notebooks.

## Codebase Structure

### Folders
- `epsilon_transformers`: source code for this repository.

### Code
- `run_sweeps.py`: run sweeps on wandb

## Usage

To install, alongside all dependencies, run `pip install -e .` from the repository folder

## Persistence

This codebase provides functionality to persist models & other research artifact's to the S3 cloud provider. In order to use it, make sure that you have the `AWS_ACCESS_KEY_ID` `AWS_SECRET_ACCESS_KEY` environment variables set in a .env file.

If you are a PIBBSS affiliate, or otherwise feel like you should have access to PIBBSS' S3 instance but currently don't, reach out to Lucas.

## Data Available on S3

### Model Checkpoints
Model checkpoints are saved as `.pt` files, of the form `{number_of_tokens_trained}.pt`.

### Log Data
Log data is saved as `.csv` files, of the form `{train_log}.csv` and `{val_log}.csv`. The columns of these `.csv` files are as follows:

- `_timestamp`: Timestamp of the logged entry
- `_step`: Row of log in wandb
- `_runtime`: Runtime of the logged entry
- `loss`: Training loss value
- `val_loss`: Validation loss value
- `relative_loss_[0-L]`: Relative loss values for different context window positions (e.g., `relative_loss_0`, `relative_loss_1`, ..., `relative_loss_L`), where `L` is the context window length. Relative loss means the loss relative to the optimal loss at that position.
- `val_relative_loss_[0-L]`: Validation relative loss values for different context window positions (e.g., `val_relative_loss_0`, `val_relative_loss_1`, ..., `val_relative_loss_L`), where `L` is the context window length. Relative loss means the loss relative to the optimal loss at that position.
- `tokens_trained`: the total number of tokens trained on up to that point

### Training Configuration
Training configuration is saved as a JSON file named `train_config.json`. The file contains the following parameters:

- seed: Random seed used for reproducibility
- n_ctx: Context window length
- act_fn: Activation function used in the model
- d_head: Dimension of each attention head
- d_model: Dimension of the model's hidden states
- d_vocab: Size of the vocabulary
- n_heads: Number of attention heads
- n_iters: Number of iterations per epoch
- n_layers: Number of transformer layers
- optimizer: Optimizer used for training
- batch_size: Batch size used during training
- num_epochs: Total number of training epochs
- weight_decay: Weight decay factor for regularization
- learning_rate: Learning rate used for optimization
- normalization_type: Type of normalization used in the model

### Specific Traning Runs
- rrxor - https://wandb.ai/adamimos/transformer-MSPs/runs/vfs4q106/
- mess3


## Dev

For formatting, type checking, and testing you can run the corresponding script in `scripts/`
