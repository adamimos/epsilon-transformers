# epsilon-transformers/config/config.py

import itertools

# Define hyperparameter ranges
hyperparams = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'n_layers': [1, 2, 3],
    'd_model': [32, 64],
    'n_heads': [4, 8],
}

# Generate all combinations of hyperparameters
def generate_hyperparam_combinations(hyperparams_dict):
    keys = hyperparams_dict.keys()
    values = hyperparams_dict.values()
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))