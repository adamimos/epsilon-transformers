# Epsilon Transformers

This project contains Python code for generating process data and training transformer models on it. The codebase is organized into several Python scripts and Jupyter notebooks.

## Codebase Structure

### Folders
- `epsilon_transformers`: source code for this repository.

### Code
- `run_sweeps.py`: run sweeps on wandb

## Usage

We use poetry as our dependency management.

First, [follow the following instructions](https://python-poetry.org/docs/#installation) to make sure that you have poetry installed

Second, you can install all dependencies by running
```poetry install```

Third, activate the approriate venv by running
```poetry shell```

and you should be good to go.

If you want to run one of the jupyter notebooks in `examples/` you'll have to make sure you are pointing to the python interpreter which corresponds to your venv. You check the path of the python interpreter by running
```which python```
while you are correct shell activated

