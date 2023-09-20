# %% Imports

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
import time
import h5py


# Import necessary functions from the modules
from markov_utilities import (
    create_random_epsilon_machine, 
    generate_emissions,
    calculate_steady_state_distribution
)
from visualization import (
    visualize_graph_with_selective_offset, 
    plot_block_entropy_diagram, 
    plot_conditional_entropy_diagram
)
from entropy_analysis import (
    compute_block_entropy, 
    compute_conditional_entropy, 
    compute_empirical_conditional_entropy
)

from error_analysis import (
    compute_minimum_error,
    inverse_binary_entropy,
    binary_entropy
)

from data_persistence import save_epsilon_machine_to_file

# magic for autoreloading modules when they change
%load_ext autoreload
%autoreload 2



# %% Count percentage of existing states for different num_states
# the original paper says ~80%

NUM_STATES = [30, 300, 3000]
NUM_SYMBOLS = 2  # binary strings
ALPHA = 1.0

# Store the average percentages for each num_states
average_percentages = []

for num_states in NUM_STATES:
    states_ratio = []  # List to store the ratio of states that exist to the initial num_states
    for _ in tqdm(range(1000), desc=f"Generating machines for {num_states} states"):
        _, epsilon_machine = create_random_epsilon_machine(num_states, NUM_SYMBOLS, ALPHA)
        actual_states = epsilon_machine[0].shape[0]
        states_ratio.append(actual_states / num_states)
    
    avg_percentage = 100 * sum(states_ratio) / len(states_ratio)
    average_percentages.append(avg_percentage)

# Convert the results into a DataFrame
df = pd.DataFrame({
    'Initial Num of States': NUM_STATES,
    'Average Percentage of Existing States': average_percentages
})

# Display the results using tabulate
print(tabulate(df, headers='keys', tablefmt='grid'))


# %% CREATE DATA FOR BENCHMARKS

NUM_SYMBOLS = 2
ALPHA = 1.0
NUM_STATES_LIST = [30, 300, 3000]
NUM_REPEATS = 10
NUM_EMISSIONS = 1000000

# File path
hdf5_filepath_base = 'data/results.h5'

for num_states in NUM_STATES_LIST:
    hdf5_filepath = hdf5_filepath_base.replace('.h5', f'_{num_states}_states.h5')

    with h5py.File(hdf5_filepath, 'a') as f: # 'a' means read/write if exists, create otherwise

        for repeats in tqdm(range(NUM_REPEATS)):
            group_name = f"repeat_{repeats}"
            if group_name in f: # Check if results for this configuration already exist
                continue
            
            # create the epsilon machine and emissions
            _, epsilon_machine = create_random_epsilon_machine(num_states, NUM_SYMBOLS, ALPHA)
            emissions = generate_emissions(epsilon_machine, NUM_EMISSIONS)
            conditional_entropies = compute_empirical_conditional_entropy(emissions, 15)

            # Create group for this configuration
            group = f.create_group(group_name)
            group.create_dataset('epsilon_machine', data=epsilon_machine, compression="gzip", compression_opts=9)
            group.create_dataset('emissions', data=emissions, dtype='uint8')

            # Create dataset for results within the group
            results = []
            for block_size, cond_entropy in enumerate(conditional_entropies, 1):
                inverse_entropy = inverse_binary_entropy(cond_entropy)
                min_error = compute_minimum_error(epsilon_machine)
                error_ratio = (inverse_entropy - min_error) / min_error
                results.append({
                    "Block Size": block_size,
                    "Conditional Entropy": cond_entropy,
                    "Inverse Binary Entropy": inverse_entropy,
                    "Minimum Error": min_error,
                    "Error Ratio": error_ratio
                })

            results_df = pd.DataFrame(results)
            rec_array = results_df.to_records(index=False)
            group.create_dataset('results', data=rec_array)



# %%
# Constants
NUM_STATES_LIST = [30, 300, 3000]
hdf5_filepath_base = 'data/results.h5'

# Create a list to store all the results
all_results = []

for num_states in NUM_STATES_LIST:
    hdf5_filepath = hdf5_filepath_base.replace('.h5', f'_{num_states}_states.h5')
    
    with h5py.File(hdf5_filepath, 'r') as f: # 'r' means read-only
        
        for group_name in f:
            # Extract the results dataset and convert to DataFrame
            results_array = f[group_name]['results'][:]
            df = pd.DataFrame(results_array)
            df['Number of States'] = num_states
            all_results.append(df)

# Concatenate all the dataframes
final_df = pd.concat(all_results, ignore_index=True)

# Plot the error ratio vs block size
plt.figure(figsize=(10, 6))
sns.lineplot(data=final_df, x='Block Size', y='Error Ratio', hue='Number of States',
              palette='bright')
plt.title('Error Ratio vs Block Size for Different State Counts')
plt.grid(True)
plt.show()

# %%
