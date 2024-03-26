# Epsilon Transformers

This project contains Python code for generating process data and training transformer models on it. The codebase is organized into several Python scripts and Jupyter notebooks.

## Codebase Structure

### Folders
- `epsilon_transformers`: source code for this repository.

### Code
- `run_sweeps.py`: run sweeps on wandb

## Usage

To install, alongside all dependencies, run `pip install -e .` from the repository folder

## Dev

For formatting, type checking, and testing you can run the corresponding script in `scripts/`

## SplineCam

This repo uses the splinecam as a submodule, and setting this up takes a couple of steps.
(Note, these set up steps are specific to Linux)

1. Make sure you use `git clone --recursive https://github.com/adamimos/epsilon-transformers.git` when you clone this repo in order to clone the submodule as well. Alternatively, you can also use run `git submodule update --init --recursive` after cloning the repo.

2. You're going to want to create a virtual environment for your submodule. 
`cd splinecam`
`python -m venv splinecam_env`
`source splinecam_env/bin/activate`

3. Follow all of the install instructions on the splinecam README
(You may have to give a sudo depending on your machine)
(Also, in the middle of the first `apt-get install` I was asked to put in my location... sus. I just put in random things and I think it worked fine)

4. Now, since `graph_tool` is a system-wide package, we're going to have to create a symlink for it inside of our virtual environment.

First, find the location of your `graph_tool` package with
`dpkg -L python3-graph-tool` this should list all the files which were affected when you installed `graph-tool` via `apt-get install`. You're going to want to find the top level directory for your package in your system-wide lib. That will most likely be `/usr/lib/python3/dist-packages/graph_tool`.

And now to create your simlink run
`ln -s /usr/lib/python3/dist-packages/graph_tool splinecam_env/lib/pythonX.Y/site-packages/`
(where X.Y are the appropraite python version numbers)

et voila! You should be ready to go.
