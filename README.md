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

## Dev

For formatting, type checking, and testing you can run the corresponding script in `scripts/`
