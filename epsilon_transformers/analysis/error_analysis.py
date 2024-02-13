from scipy.optimize import minimize_scalar
from typing import List
import numpy as np
from epsilon_transformers.markov_utilities import calculate_steady_state_distribution

NUM_SYMBOLS = 2

def compute_minimum_error(epsilon_machine: np.ndarray) -> float:
    """Compute the minimum error for predicting the next symbol based on the epsilon machine."""
    # Compute the steady state distribution for the epsilon machine
    steady_state_distribution = calculate_steady_state_distribution(sum(epsilon_machine))
    
    # Initialize the minimum error
    min_error = 0
    
    # Iterate over each state
    for state, state_prob in enumerate(steady_state_distribution):
        # For each state, get the maximum emission probability
        max_emission_prob = max([np.sum(epsilon_machine[emission][state, :]) for emission in range(NUM_SYMBOLS)])
        # Update the minimum error based on the state's contribution
        min_error += state_prob * (1 - max_emission_prob)
    
    return min_error

def binary_entropy(p: float) -> float:
    """Compute the binary entropy for a given probability p."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def inverse_binary_entropy(target_entropy: float) -> float:
    """Find the probability p corresponding to a given binary entropy value."""
    # Objective function: the difference between target entropy and binary entropy of p
    objective = lambda p: (binary_entropy(p) - target_entropy)**2
    
    # Minimize the objective function to find p
    result = minimize_scalar(objective, bounds=(0, 0.5), method='bounded')
    
    return result.x

