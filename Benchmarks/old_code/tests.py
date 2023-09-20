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

# %% Test and visualize the random epsilon machine

# Constants
NUM_STATES = 10
NUM_SYMBOLS = 2
ALPHA = 0.1

# Test and visualize the random epsilon machine
H, matrices = create_random_epsilon_machine(NUM_STATES, NUM_SYMBOLS, ALPHA)
visualize_graph_with_selective_offset(H)

# %%

NUM_STATES_LIST = [30, 300, 3000]
NUM_SYMBOLS = 2  # binary strings
ALPHA = 1.0

# Initialize results dictionary
timing_results = {
    'num_states': [],
    'epsilon_machine_time': [],
    'emissions_time': []
}

for num_states in tqdm(NUM_STATES_LIST):
    # Measure time taken for creating epsilon machine
    start_time = time.time()
    _, epsilon_machine = create_random_epsilon_machine(num_states, NUM_SYMBOLS, ALPHA)
    epsilon_machine_time = time.time() - start_time
    
    # Measure time taken for generating emissions
    start_time = time.time()
    emissions = generate_emissions(epsilon_machine, 100000)
    emissions_time = time.time() - start_time
    
    # Store results
    timing_results['num_states'].append(num_states)
    timing_results['epsilon_machine_time'].append(epsilon_machine_time)
    timing_results['emissions_time'].append(emissions_time)

# Display results using a DataFrame for better visualization
import pandas as pd
df = pd.DataFrame(timing_results)
print(tabulate(df, headers='keys', tablefmt='grid'))

# %%
# Create and test epsilon machine
epsilon_machine_graph, epsilon_machine = create_random_epsilon_machine(300, NUM_SYMBOLS, ALPHA)
emissions = generate_emissions(epsilon_machine, 100000)

# Generate the block entropy diagram for the generated sequence
plot_block_entropy_diagram(emissions, max_block_length=15)
# Generate the conditional entropy diagram for the current sequence
plot_conditional_entropy_diagram(emissions, max_block_length=15)


# %%
# Gather block entropies for multiple runs
NUM_STATES = 300
NUM_SYMBOLS = 2
ALPHA = 1.0

all_block_entropies = []
for _ in tqdm(range(1000)):
    _, epsilon_machine = create_random_epsilon_machine(NUM_STATES, NUM_SYMBOLS, ALPHA)
    emissions = generate_emissions(epsilon_machine, 10000)
    block_entropies = compute_block_entropy(emissions, max_block_length=15)
    all_block_entropies.append(block_entropies)
# %%

# Gather conditional entropies for multiple states and runs
results = {}
for repeats in tqdm(range(10)):
    for num_states in [30, 300, 3000]:
        if num_states not in results:
            results[num_states] = []
        _, epsilon_machine = create_random_epsilon_machine(num_states, NUM_SYMBOLS, ALPHA)
        emissions = generate_emissions(epsilon_machine, 1000000)
        conditional_entropy = compute_empirical_conditional_entropy(emissions, 15)
        results[num_states].append(conditional_entropy)
# %%
# Convert results to a DataFrame and plot
data = [
    [num_states, repeat_num, block_len, entropy_val]
    for num_states, entropies in results.items()
    for repeat_num, repeat_entropies in enumerate(entropies)
    for block_len, entropy_val in enumerate(repeat_entropies)
]
df = pd.DataFrame(data, columns=['num_states', 'repeat_number', 'block_length', 'conditional_entropy'])
sns.lineplot(data=df, x='block_length', y='conditional_entropy', hue='num_states',
              errorbar=('ci', 90), legend='full', palette='bright')
plt.ylim([0.3, 0.7])
plt.show()