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
    save_nn_data
)

import torch
import sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiprocessing import Pool
import os


def analyze_single_run(args):
    """Function to analyze a single run (to be called in parallel)"""
    sweep_id, run = args
    loader = S3ModelLoader()  # Create new loader instance for each process
    sweep_config = loader.load_sweep_config(sweep_id)
    print(f"Analyzing {run}")
    
    ckpts = loader.list_checkpoints(sweep_id, run)
    
    # Load initial model and prepare data
    model, config = loader.load_checkpoint(
        sweep_id=sweep_id,
        run_id=run,
        checkpoint_key=ckpts[-1],
        device='cpu'
    )

    nn_type = model_type(model)
    
    nn_data = prepare_msp_data(config, config['model_config'])
    (nn_inputs, nn_beliefs, nn_belief_indices, 
     nn_probs, nn_unnormalized_beliefs) = nn_data
    
    # Save the data in a dictionary format
    data_to_save = {
        'inputs': nn_inputs,
        'beliefs': nn_beliefs,
        'belief_indices': nn_belief_indices,
        'probs': nn_probs,
        'unnormalized_beliefs': nn_unnormalized_beliefs
    }
    
    nn_shuffled_beliefs = shuffle_belief_norms(nn_unnormalized_beliefs)
    data_to_save['shuffled_beliefs'] = nn_shuffled_beliefs

    markov_data = markov_approx_msps(config, max_order=3)
    for order, mark_data in enumerate(markov_data):
        mark_inputs, mark_beliefs, mark_indices, mark_probs, mark_unnorm = mark_data
        mark_shuffled = shuffle_belief_norms(mark_unnorm)
        
        data_to_save.update({
            f'markov_order_{order}_inputs': mark_inputs,
            f'markov_order_{order}_beliefs': mark_beliefs,
            f'markov_order_{order}_indices': mark_indices,
            f'markov_order_{order}_probs': mark_probs,
            f'markov_order_{order}_unnormalized': mark_unnorm,
            f'markov_order_{order}_shuffled': mark_shuffled
        })

    print(f'Run {run}: data size = {sys.getsizeof(data_to_save)/1024**2} MB')
    save_nn_data(loader, sweep_id, run, data_to_save)

    # Analyze checkpoints
    for ckpt in ckpts:

        is_final_ckpt = ckpt == ckpts[-1]

        model, config = loader.load_checkpoint(
            sweep_id=sweep_id,
            run_id=run,
            checkpoint_key=ckpt,
            device='cpu'
        )
        sweep_type = get_sweep_type(run)

        # Analyze normalized beliefs
        analyze_model_checkpoint(
            model, nn_inputs, nn_type, nn_beliefs, 
            nn_belief_indices, nn_probs, sweep_type, run, title="Normalized Beliefs",
            loader=loader,
            checkpoint_key=ckpt,
            sweep_id=sweep_id,
            save_figure = is_final_ckpt
        )

        # Analyze unnormalized beliefs
        analyze_model_checkpoint(
            model, nn_inputs, nn_type, nn_unnormalized_beliefs, 
            nn_belief_indices, nn_probs, sweep_type, run, title="Unnormalized Beliefs",
            loader=loader,
            checkpoint_key=ckpt,
            sweep_id=sweep_id,
            save_figure = is_final_ckpt
        )

        # Analyze shuffled unnormalized beliefs
        analyze_model_checkpoint(
            model, nn_inputs, nn_type, nn_shuffled_beliefs, 
            nn_belief_indices, nn_probs, sweep_type, run, title="Shuffled Unnormalized Beliefs",
            loader=loader,
            checkpoint_key=ckpt,
            sweep_id=sweep_id,
            save_figure = is_final_ckpt
        )

        # Analyze markov approximations
        for order, mark_data in enumerate(markov_data):
            # unpack the data
            nn_inputs, nn_beliefs, nn_belief_indices, nn_probs, nn_unnormalized_beliefs = mark_data
             
            # Create shuffled version of unnormalized beliefs
            nn_shuffled_beliefs = shuffle_belief_norms(nn_unnormalized_beliefs)
            
            # Analyze normalized beliefs
            analyze_model_checkpoint(
                model, nn_inputs, nn_type, nn_beliefs, 
                nn_belief_indices, nn_probs, sweep_type, run, title=f"Order-{order} Approx.",
                loader=loader,
                checkpoint_key=ckpt,
                sweep_id=sweep_id,
                save_figure = is_final_ckpt
            )
             
            # Analyze unnormalized beliefs
            analyze_model_checkpoint(
                model, nn_inputs, nn_type, nn_unnormalized_beliefs, 
                nn_belief_indices, nn_probs, sweep_type, run, title=f"Order-{order} Approx. Unnormalized",
                loader=loader,
                checkpoint_key=ckpt,
                sweep_id=sweep_id,
                save_figure = is_final_ckpt
            )

            # Analyze shuffled unnormalized beliefs
            analyze_model_checkpoint(
                model, nn_inputs, nn_type, nn_shuffled_beliefs, 
                nn_belief_indices, nn_probs, sweep_type, run, title=f"Order-{order} Approx. Shuffled Unnormalized",
                loader=loader,
                checkpoint_key=ckpt,
                sweep_id=sweep_id,
                save_figure = is_final_ckpt
            )

    return f"Completed analysis for {run}"

def main():
    sweeps = {
        '20241121152808': 'RNN',
        '20241205175736': 'Transformer'
    }

    # Create list of all (sweep_id, run) pairs to analyze
    all_tasks = []
    for sweep_id in sweeps:
        loader = S3ModelLoader()
        runs = loader.list_runs_in_sweep(sweep_id)
        all_tasks.extend([(sweep_id, run) for run in runs])

    # Number of CPUs to use (leave some cores free for system)
    n_processes = max(1, os.cpu_count() - 2)
    print(f"Using {n_processes} processes")

    # Run analyses in parallel
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(analyze_single_run, all_tasks),
            total=len(all_tasks)
        ))

    # Print results
    for result in results:
        print(result)

if __name__ == '__main__':
    main()


# %%
