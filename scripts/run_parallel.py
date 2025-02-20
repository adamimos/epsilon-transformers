# %%
# launch_runs.py
import subprocess
from epsilon_transformers.analysis.load_data import S3ModelLoader

def run_process(run_id):
    command = ["python", "./scripts/run_single_run.py", run_id]
    print(f"Launching run {run_id}...")
    subprocess.run(command)

def main():
    loader = S3ModelLoader()
    sweep_id = '20241205175736'
    start_ind = 0

    # Get all run IDs
    all_run_ids = loader.list_runs_in_sweep(sweep_id)[start_ind:]
    total_runs = len(all_run_ids)

    print(f"Total runs to process: {total_runs}")
    print("Running serially...")

    # Process runs one at a time
    for run_id in all_run_ids:
        run_process(run_id)

if __name__ == '__main__':
    main()
# %%
