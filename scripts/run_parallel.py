# %%
# launch_runs.py
import subprocess
from epsilon_transformers.analysis.load_data import S3ModelLoader
import multiprocessing as mp
import os
from math import ceil

def run_process(run_id):
    command = ["python", "./scripts/run_single_run.py", run_id]
    print(f"Launching run {run_id} in a new process...")
    subprocess.run(command)

def main():
    loader = S3ModelLoader()
    sweep_id = '20241205175736'
    start_ind = 0

    # Get all run IDs
    all_run_ids = loader.list_runs_in_sweep(sweep_id)[start_ind:]
    total_runs = len(all_run_ids)

    # Calculate optimal number of cores to use
    available_cores = mp.cpu_count()
    # Use min of: total runs, available cores, and 75% of available cores
    optimal_cores = min(
        total_runs,  # Don't use more cores than runs
        available_cores,  # Don't exceed available cores
        max(1, ceil(available_cores * 0.75))  # Stay within 75% limit
    )

    print(f"Total runs to process: {total_runs}")
    print(f"Available cores: {available_cores}")
    print(f"Using {optimal_cores} cores for processing")

    # Create a pool of workers and map the run_process function to run_ids
    with mp.Pool(optimal_cores) as pool:
        pool.map(run_process, all_run_ids)

if __name__ == '__main__':
    main()
# %%
