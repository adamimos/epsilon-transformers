#%%

from epsilon_transformers.process.GHMM import markov_approximation
from epsilon_transformers.analysis.load_data import S3ModelLoader
from epsilon_transformers.analysis.activation_analysis import (
    prepare_msp_data,
    run_activation_to_beliefs_regression,
    get_sweep_type,
    model_type,
    get_activations,
    plot_belief_prediction_comparison,
    analyze_all_layers,
    analyze_model_checkpoint,
    markov_approx_msps,
    shuffle_belief_norms,
    save_nn_data,
    get_process_filename,
    load_nn_data,
    ProcessDataLoader

)
import time

import torch
import sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import os
import json
import numpy as np
import argparse


def check_run_completed(loader: S3ModelLoader, sweep_id: str, run_id: str) -> bool:
    """Check if a run has already been analyzed by looking for a completion marker."""
    try:
        loader.s3_client.head_object(
            Bucket=loader.bucket_name,
            Key=f"analysis/{sweep_id}/{run_id}/analysis_complete.json"
        )
        print(f"Run {run_id} already analyzed")
        return True
    except loader.s3_client.exceptions.ClientError:
        return False

def mark_run_completed(loader: S3ModelLoader, sweep_id: str, run_id: str):
    """Mark a run as completed by creating a completion marker file."""
    completion_data = {
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sweep_id': sweep_id,
        'run_id': run_id
    }
    
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=f"analysis/{sweep_id}/{run_id}/analysis_complete.json",
        Body=json.dumps(completion_data)
    )
    print(f"Marked run {run_id} as completed")

def check_checkpoint_completed(loader: S3ModelLoader, sweep_id: str, run_id: str, checkpoint_key: str) -> bool:
    """Check if a checkpoint analysis has been completed."""
    try:
        # Extract checkpoint number from key
        checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
        
        # Check for completion marker
        loader.s3_client.head_object(
            Bucket=loader.bucket_name,
            Key=f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/completed.json"
        )
        print(f"Checkpoint {checkpoint_num} already analyzed")
        return True
    except loader.s3_client.exceptions.ClientError:
        return False

def mark_checkpoint_completed(loader: S3ModelLoader, sweep_id: str, run_id: str, checkpoint_key: str):
    """Mark a checkpoint as completed."""
    checkpoint_num = checkpoint_key.split('/')[-1].replace('.pt', '')
    completion_data = {
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sweep_id': sweep_id,
        'run_id': run_id,
        'checkpoint': checkpoint_num
    }
    
    loader.s3_client.put_object(
        Bucket=loader.bucket_name,
        Key=f"analysis/{sweep_id}/{run_id}/checkpoint_{checkpoint_num}/completed.json",
        Body=json.dumps(completion_data)
    )
    print(f"Marked checkpoint {checkpoint_num} as completed")

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model data in parallel')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['rnn', 'transformer', 'all'],
        default='all',
        help='Type of model to analyze (rnn, transformer, or all)'
    )
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Process checkpoints in reverse order'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define all sweeps
    sweeps = {
        '20241121152808': 'RNN',
        '20241205175736': 'Transformer'
    }
    
    # Filter sweeps based on model type argument
    if args.model_type != 'all':
        sweeps = {
            sweep_id: model_type 
            for sweep_id, model_type in sweeps.items() 
            if model_type.lower() == args.model_type.lower()
        }
    
    if not sweeps:
        print(f"No sweeps found for model type: {args.model_type}")
        return

    # Get number of available CPUs, leaving some headroom
    num_cpus = cpu_count()
    num_workers = max(1, num_cpus - 2)  # Leave 2 CPUs free for system tasks
    print(f"Using {num_workers} workers out of {num_cpus} available CPUs")

    for sweep_id in sweeps:
        loader = S3ModelLoader()
        runs = loader.list_runs_in_sweep(sweep_id)
        
        # Filter out already completed runs
        runs_to_process = [
            run for run in runs 
            if not check_run_completed(loader, sweep_id, run)
        ]
        
        if not runs_to_process:
            print(f"All runs in sweep {sweep_id} already processed")
            continue

        # Prepare arguments for parallel processing
        run_args = [(sweep_id, run) for run in runs_to_process]
        
        # Process runs in parallel
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(analyze_single_run, run_args),
                total=len(run_args),
                desc=f"Processing sweep {sweep_id}"
            ))

        # Mark successful runs as completed
        for run in runs_to_process:
            try:
                mark_run_completed(loader, sweep_id, run)
            except Exception as e:
                print(f"Error marking run {run} as completed: {str(e)}")

        if hasattr(loader, 'async_uploader'):
            loader.async_uploader.shutdown()

def analyze_single_run(args):
    """Function to analyze a single run"""
    sweep_id, run = args
    start_time = time.time()
    
    try:
        print(f"\nStarting analysis for run {run}")
        loader = S3ModelLoader()
        process_loader = ProcessDataLoader(loader)

        # Load initial model and config 
        load_start = time.time()
        model, config = loader.load_checkpoint(sweep_id, run, loader.list_checkpoints(sweep_id, run)[-1], device='cpu')
        print(f"Initial model loading took {time.time() - load_start:.2f}s")

        # Load or generate process data
        data_start = time.time()
        base_data, markov_data = process_loader.load_or_generate_process_data(sweep_id, run, model, config)
        print(f"Process data preparation took {time.time() - data_start:.2f}s")

        # unpack base data
        nn_inputs = base_data['inputs']
        nn_beliefs = base_data['beliefs']
        nn_belief_indices = base_data['belief_indices']
        nn_probs = base_data['probs']
        nn_unnormalized_beliefs = base_data['unnormalized_beliefs']
        nn_shuffled_beliefs = base_data['shuffled_beliefs']
 
        process_config = config['process_config']
        process_folder_name = get_process_filename(process_config)
        ckpts = loader.list_checkpoints(sweep_id, run)
        if parse_args().reverse:  # Get args in the worker process
            print("Reversing checkpoints")
            ckpts = ckpts[::-1]  # Reverse the list
        nn_type = model_type(model)

        # Filter out already completed checkpoints
        ckpts_to_process = [
            ckpt for ckpt in ckpts 
            if not check_checkpoint_completed(loader, sweep_id, run, ckpt)
        ]

        # Process each checkpoint
        for ckpt_ind, ckpt in enumerate(tqdm(ckpts_to_process, desc=f"Processing checkpoints for run {run}")):
            try:
                ckpt_start = time.time()
                
                # Load model for this checkpoint
                model, config = loader.load_checkpoint(
                    sweep_id=sweep_id,
                    run_id=run,
                    checkpoint_key=ckpt,
                    device='cpu'
                )
                
                sweep_type = get_sweep_type(run)
                
                # Define analyses to run
                analyses = [
                    (nn_inputs, nn_beliefs, "Normalized Beliefs"),
                    (nn_inputs, nn_unnormalized_beliefs, "Unnormalized Beliefs"),
                    (nn_inputs, nn_shuffled_beliefs, "Shuffled Unnormalized Beliefs")
                ]
                
                # Add Markov analyses
                for order, mark_data in enumerate(markov_data):
                    # Unpack the tuple, handling both 5 and 6 value cases
                    if len(mark_data) == 6:
                        mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm, mark_shuffled = mark_data
                    else:
                        mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm = mark_data
                        mark_shuffled = shuffle_belief_norms(mark_unnorm)
                        
                    analyses.extend([
                        (mark_inputs, mark_beliefs, f"Order-{order} Approx."),
                        (mark_inputs, mark_unnorm, f"Order-{order} Approx. Unnormalized"),
                        (mark_inputs, mark_shuffled, f"Order-{order} Approx. Shuffled Unnormalized")
                    ])
                
                # Run analyses for this checkpoint
                for i, (inputs, beliefs, title) in enumerate(analyses):
                    analysis_start = time.time()
                    analyze_model_checkpoint(
                        model=model,
                        nn_inputs=inputs,
                        nn_type=nn_type,
                        nn_beliefs=beliefs,
                        nn_belief_indices=nn_belief_indices,
                        nn_probs=nn_probs,
                        sweep_type=sweep_type,
                        run_name=run,
                        sweep_id=sweep_id,
                        title=title,
                        loader=loader,
                        checkpoint_key=ckpt,
                        save_figure=True
                    )
                    print(f"Analysis {i+1}/{len(analyses)} ({title}) took {time.time() - analysis_start:.2f}s")
                
                mark_checkpoint_completed(loader, sweep_id, run, ckpt)
                print(f"Checkpoint {ckpt_ind} took {time.time() - ckpt_start:.2f}s")
                
            except Exception as e:
                print(f"Error processing checkpoint {ckpt} for run {run}, title {title}: {str(e)}")
                continue

        print(f"Total run time for {run}: {time.time() - start_time:.2f}s")
        return f"Completed analysis for {run}"
        
    except Exception as e:
        print(f"Error processing run {run}: {str(e)}")
        return f"Failed to process run {run}: {str(e)}"

if __name__ == '__main__':
    main()


# %%
