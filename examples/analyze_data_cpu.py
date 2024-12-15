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
from multiprocessing import Pool
import os


def main():
    sweeps = {
        '20241121152808': 'RNN',
        '20241205175736': 'Transformer'
    }

    # Process one sweep at a time
    for sweep_id in sweeps:
        loader = S3ModelLoader()
        runs = loader.list_runs_in_sweep(sweep_id)
        
        # Process one run at a time
        for run in runs:
            print(f"\nProcessing sweep {sweep_id}, run {run}")
            analyze_single_run((sweep_id, run))

def analyze_checkpoint(args):
    """Function to analyze a single checkpoint (to be called in parallel)"""
    ckpt_ind, model, nn_inputs, nn_type, nn_beliefs, nn_belief_indices, nn_probs, sweep_type, run, sweep_id, title, ckpt, is_final_ckpt = args
    print(f"Analyzing ckpt {ckpt_ind} {title}")

    # Create new loader instance for this process
    loader = S3ModelLoader()
    
    print(f"initiating analysis for {title}")
    # Analyze model checkpoint with the prepared data
    analyze_model_checkpoint(
        model=model,
        nn_inputs=nn_inputs,
        nn_type=nn_type,
        nn_beliefs=nn_beliefs,
        nn_belief_indices=nn_belief_indices,
        nn_probs=nn_probs,
        sweep_type=sweep_type,
        run_name=run,
        sweep_id=sweep_id,
        title=title,
        loader=loader,
        checkpoint_key=ckpt,
        save_figure=is_final_ckpt
    )
    
    return f"Completed analysis for {run} checkpoint {ckpt}"

def analyze_single_run(args):
    """Function to analyze a single run, parallelizing across checkpoints"""
    sweep_id, run = args
    loader = S3ModelLoader()
    process_loader = ProcessDataLoader(loader)

    # Load initial model and config 
    model, config = loader.load_checkpoint(sweep_id, run, loader.list_checkpoints(sweep_id, run)[-1], device='cpu')

    base_data, markov_data = process_loader.load_or_generate_process_data(sweep_id, run, model, config)

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
    nn_type = model_type(model)

    for order in range(len(markov_data)):
        mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm = markov_data[order]
        mark_shuffled = shuffle_belief_norms(mark_unnorm)
        # Update the tuple in markov_data
        markov_data[order] = (mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm, mark_shuffled)

    # Prepare arguments for parallel checkpoint analysis
    checkpoint_args = []
    for ckpt_ind, ckpt in enumerate(tqdm(ckpts)):
        is_final_ckpt = True
        
        model, config = loader.load_checkpoint(
            sweep_id=sweep_id,
            run_id=run,
            checkpoint_key=ckpt,
            device='cpu'
        )
        sweep_type = get_sweep_type(run)
        
        # Add arguments for different types of analysis
        # Normal beliefs
        checkpoint_args.append(( ckpt_ind,
            model, nn_inputs, nn_type, nn_beliefs, 
            nn_belief_indices, nn_probs, sweep_type,
            run, sweep_id, "Normalized Beliefs",
            ckpt, is_final_ckpt  # Removed loader from args
        ))
        
        # Unnormalized beliefs
        checkpoint_args.append(( ckpt_ind,
            model, nn_inputs, nn_type, nn_unnormalized_beliefs,
            nn_belief_indices, nn_probs, sweep_type,
            run, sweep_id, "Unnormalized Beliefs",
            ckpt, is_final_ckpt  # Removed loader from args
        ))
        
        # Shuffled beliefs
        checkpoint_args.append(( ckpt_ind,
            model, nn_inputs, nn_type, nn_shuffled_beliefs,
            nn_belief_indices, nn_probs, sweep_type,
            run, sweep_id, "Shuffled Unnormalized Beliefs",
            ckpt, is_final_ckpt  # Removed loader from args
        ))
        
        # Markov approximations
        for order, mark_data in enumerate(markov_data):
            mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm, mark_shuffled = mark_data
            
            # Normal Markov
            checkpoint_args.append(( ckpt_ind,
                model, mark_inputs, nn_type, mark_beliefs,
                mark_indices, mark_probs, sweep_type,
                run, sweep_id, f"Order-{order} Approx.",
                ckpt, is_final_ckpt  # Removed loader from args
            ))
            
            # Unnormalized Markov
            checkpoint_args.append(( ckpt_ind,
                model, mark_inputs, nn_type, mark_unnorm,
                mark_indices, mark_probs, sweep_type,
                run, sweep_id, f"Order-{order} Approx. Unnormalized",
                ckpt, is_final_ckpt  # Removed loader from args
            ))
            
            # Shuffled Markov
            checkpoint_args.append(( ckpt_ind,
                model, mark_inputs, nn_type, mark_shuffled,
                mark_indices, mark_probs, sweep_type,
                run, sweep_id, f"Order-{order} Approx. Shuffled Unnormalized",
                ckpt, is_final_ckpt  # Removed loader from args
            ))
    
    # Use multiprocessing to analyze checkpoints in parallel
    n_processes = min(10, os.cpu_count() - 2)
    print(f"Using {n_processes} processes for checkpoint analysis")

    batch_size = 4
    checkpoint_batches = [checkpoint_args[i:i+batch_size] for i in range(0, len(checkpoint_args), batch_size)]
    
    with Pool(processes=n_processes) as pool:
        for batch in checkpoint_batches:
            results = list(tqdm(
                pool.imap_unordered(analyze_checkpoint, batch),
                total=len(batch),
                desc=f"Analyzing checkpoints for {run}"
            ))

            time.sleep(.1)

    # Print results
    for result in results:
        print(result)

    return f"Completed analysis for {run}"

if __name__ == '__main__':
    main()


# %%
