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

def check_checkpoint_completed_wrapper(args):
    """Wrapper function that creates its own S3 client"""
    sweep_id, run_id, checkpoint_key = args
    loader = S3ModelLoader()  # Create new loader instance in this process
    return check_checkpoint_completed(loader, sweep_id, run_id, checkpoint_key)

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
        print(f"Checkpoint {checkpoint_num} of {run_id} already analyzed")
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
    parser.add_argument(
        '--num-cpus',
        type=int,
        default=None,
        help='Number of CPU workers to use. Defaults to number of CPUs minus 2'
    )
    return parser.parse_args()

def analyze_single_checkpoint(args):
    """Function to analyze a single checkpoint"""
    sweep_id, run, checkpoint = args
    start_time = time.time()
    
    try:
        print(f"\nStarting analysis for checkpoint {checkpoint} of run {run}")
        loader = S3ModelLoader()
        
        # Check if checkpoint is already completed
        if check_checkpoint_completed(loader, sweep_id, run, checkpoint):
            print(f"Checkpoint {checkpoint} in {run} already completed, skipping...")
            return f"Skipped {checkpoint} (already completed)"
        
        process_loader = ProcessDataLoader(loader)

        # Load model and config
        model, config = loader.load_checkpoint(
            sweep_id=sweep_id,
            run_id=run,
            checkpoint_key=checkpoint,
            device='cpu'
        )

        # Load or generate process data
        base_data, markov_data = process_loader.load_or_generate_process_data(sweep_id, run, model, config)

        # Unpack base data
        nn_inputs = base_data['inputs']
        nn_beliefs = base_data['beliefs']
        nn_belief_indices = base_data['belief_indices']
        nn_probs = base_data['probs']
        nn_unnormalized_beliefs = base_data['unnormalized_beliefs']
        nn_shuffled_beliefs = base_data['shuffled_beliefs']

        nn_type = model_type(model)
        sweep_type = get_sweep_type(run)
        
        # Define analyses to run
        analyses = [
            (nn_inputs, nn_beliefs, "Normalized Beliefs", nn_belief_indices, nn_probs),
            (nn_inputs, nn_unnormalized_beliefs, "Unnormalized Beliefs", nn_belief_indices, nn_probs),
            (nn_inputs, nn_shuffled_beliefs, "Shuffled Unnormalized Beliefs", nn_belief_indices, nn_probs)
        ]
        
        # Add Markov analyses
        for order, mark_data in enumerate(markov_data):
            if len(mark_data) == 6:
                mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm, mark_shuffled = mark_data
            else:
                mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm = mark_data
                mark_shuffled = shuffle_belief_norms(mark_unnorm)
                
            analyses.extend([
                (mark_inputs, mark_beliefs, f"Order-{order} Approx.", mark_indices, mark_probs),
                (mark_inputs, mark_unnorm, f"Order-{order} Approx. Unnormalized", mark_indices, mark_probs),
                (mark_inputs, mark_shuffled, f"Order-{order} Approx. Shuffled Unnormalized", mark_indices, mark_probs)
            ])

        # Run analyses for this checkpoint
        for inputs, beliefs, title, indices, probs in analyses:
            analyze_model_checkpoint(
                model=model,
                nn_inputs=inputs,
                nn_type=nn_type,
                nn_beliefs=beliefs,
                nn_belief_indices=indices,
                nn_probs=probs,
                sweep_type=sweep_type,
                run_name=run,
                sweep_id=sweep_id,
                title=title,
                loader=loader,
                checkpoint_key=checkpoint,
                save_figure=True
            )
        
        mark_checkpoint_completed(loader, sweep_id, run, checkpoint)
        print(f"Checkpoint {checkpoint} took {time.time() - start_time:.2f}s")
        return f"Completed analysis for {checkpoint}"
        
    except Exception as e:
        print(f"ERROR processing checkpoint {checkpoint} for run {run}: {str(e)}")
        return f"ERROR processing checkpoint {checkpoint}: {str(e)}"

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

    # Get number of available CPUs
    if args.num_cpus is not None:
        num_workers = args.num_cpus
    else:
        num_cpus = cpu_count()
        num_workers = max(1, num_cpus - 2)
    print(f"Using {num_workers} workers out of {cpu_count()} available CPUs")


    # Collect all checkpoint tasks
    all_tasks = []
    temp_loader = S3ModelLoader()  # Temporary loader just for listing
    for sweep_id in sweeps:
        runs = temp_loader.list_runs_in_sweep(sweep_id)
        if args.reverse:
            runs = runs[::-1]

        # only take runs that are not already completed
        runs = [run for run in runs if not check_run_completed(temp_loader, sweep_id, run)]
            
        for run in runs:
            ckpts = temp_loader.list_checkpoints(sweep_id, run)
            if args.reverse:
                ckpts = ckpts[::-1]
            
            # Add each checkpoint as a task
            all_tasks.extend([(sweep_id, run, ckpt) for ckpt in ckpts])

    # Process tasks in parallel to check completion status
    with Pool(num_workers) as pool:
        try:
            completion_checks = list(pool.map(
                check_checkpoint_completed_wrapper,
                all_tasks
            ))
        except Exception as e:
            print(f"Error during completion checks: {str(e)}")
            return

    all_tasks = [task for task, is_complete in zip(all_tasks, completion_checks) if not is_complete]














    # Process checkpoints in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(analyze_single_checkpoint, all_tasks),
            total=len(all_tasks),
            desc="Processing checkpoints"
        ))

if __name__ == '__main__':
    main()


# %%
