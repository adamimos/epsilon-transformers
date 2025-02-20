# %%
# launch_runs.py
import subprocess
from epsilon_transformers.analysis.load_data import S3ModelLoader
import multiprocessing as mp
import os

def run_process(run_id):
    command = ["python", "./scripts/run_single_run.py", run_id]
    print(f"Launching run {run_id} in a new process...")
    subprocess.run(command)

def main():
    loader = S3ModelLoader()
    sweep_id = '20241205175736'
    start_ind = 0

    # Optionally, select specific run_ids.
    all_run_ids = loader.list_runs_in_sweep(sweep_id)[start_ind:]
    # For example, to run only a specific run:
    # all_run_ids = ['run_7_L1_H4_DH16_DM64_mess3']

    # Use 75% of available CPU cores
    num_cores = max(1, int(mp.cpu_count() * 0.75))
    print(f"Running on {num_cores} cores in parallel")

    # Create a pool of workers and map the run_process function to run_ids
    with mp.Pool(num_cores) as pool:
        pool.map(run_process, all_run_ids)

if __name__ == '__main__':
    main()
# %%
