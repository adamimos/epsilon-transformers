# Epsilon Transformers

This codebase contains the code for analyzing transformers from a Computational Mechanics point of view.

## Codebase Structure

The most important folders and files are:

- `epsilon_transformers`: source code for this repository.
- `examples`: examples of how to use the codebase.
    - `models`: contains saved checkpoints and log files for 3 experiments. Data must be downloaded from [this google drive](https://drive.google.com/drive/folders/1lSSkSXFS1fjsfvWIARF0qI0RHS8Be3Ja?usp=sharing), and placed in this folder. See more details in the Usage section.
    - This folder also contains python notebooks that serve as examples of the functionality of this library:
        - `compmech_process.ipynb`: Shows examples of a number of canonical Comp Mech HMMs that are available to use, and also how to instantiate a custom HMM.
        - `load_experiment_data.ipynb`: shows how to load experiment data, using RRXOR as an example
        - `compmech_rrxor.ipynb`: Comp mech analysis of the RRXOR process, including entropy rates, MSPs, and visualizations.
        - `simplex_analysis_mess3_0.05_0.85.ipynb`: MSP analysis of the mess3 experiment, recreating the fractal from the blog post.
        - `simplex_analysis_mess3_0.15_0.6.ipynb`: same as above using different hyperparameters that create a different fractal.
        - `simplex_analysis_rrxor.ipynb`: MSP analysis of the RRXOR experiment.

## Usage

### Installation
To install, alongside all dependencies, run `pip install -e .` from the repository folder.

### Downloading Experiments

Experimental data including training hyperparameters, log files of losses for both training and validation data, and checkpoints of saved models, can be found in [this google drive](https://drive.google.com/drive/folders/1lSSkSXFS1fjsfvWIARF0qI0RHS8Be3Ja?usp=sharing).

That drive contains 4 zip files:
- `models.zip` (931 MB): contains all of the data from all experiments. If you download this you shouldn't download anything else.
- `vfs4q106-rrxor.zip` (174 MB): contains the data from the RRXOR experiment.
- `f6gnm1we-mess3-0.15-0.6.zip` (196 MB): contains the data from the mess3 experiment, this is the fractal used in the blog post.
- `vfs4q106-mess3-0.05-0.85.zip` (561 MB): a mess3 experiment with different hyperparameters from above so that it is a different fractal.

You should unzip the data such that you have a folder called `models` in the `examples` folder, and inside `models` you have folders with the names of each of the experiments, and inside each of those folders you have the data.

## Dev

For formatting, type checking, and testing you can run the corresponding script in `scripts/`
