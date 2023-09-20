import numpy as np
import networkx as nx
from numpy import linalg as LA
from typing import List, Tuple


def generate_emissions(epsilon_machine: np.ndarray, num_emissions: int) -> List[int]:
    """Generate a sequence of emissions from an epsilon machine."""

    # Get the number of outputs and states
    n_outputs, n_states, _ = epsilon_machine.shape
    
    # Calculate the steady-state distribution and use it to choose the initial state
    steady_state = calculate_steady_state_distribution(epsilon_machine[0] + epsilon_machine[1])
    steady_state = steady_state / np.sum(steady_state)
    current_state = np.random.choice(n_states, p=steady_state)
    
    emissions = []
    
    # Pre-compute emission probabilities for each state
    emission_probs = np.sum(epsilon_machine, axis=2)
    
    for _ in range(num_emissions):
        # Randomly choose an emission based on available outputs and transition probabilities
        p = emission_probs[:, current_state]
        chosen_emission = np.random.choice(n_outputs, p=p / np.sum(p))
        
        # Update the current state based on the chosen emission
        current_state = np.argmax(epsilon_machine[chosen_emission, current_state, :])
        
        # Record the emission
        emissions.append(chosen_emission)
    
    return emissions

def create_markov_chain(num_states: int, num_symbols: int, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create and return a Markov chain as transition matrix and emission probabilities."""
    # Transition matrix
    transition_matrix = np.random.randint(num_states, size=(num_states, num_symbols))
    
    # Emission probabilities using a Dirichlet distribution
    emission_probabilities = np.random.dirichlet([alpha] * num_symbols, size=num_states)
    
    return transition_matrix, emission_probabilities

def transition_to_graph(transition_matrix: np.ndarray, num_symbols: int) -> nx.DiGraph:
    """Convert a transition matrix to a graph."""
    G = nx.DiGraph()
    for i in range(transition_matrix.shape[0]):
        for j in range(num_symbols):
            G.add_edge(i, transition_matrix[i, j], label=str(j))
    return G



def get_recurrent_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Extract and return the largest strongly connected component from graph G."""
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(largest_scc).copy()
    return H

def construct_transition_matrices(transition_matrix: np.ndarray, emission_probabilities: np.ndarray) -> List[np.ndarray]:
    """Construct the transition matrices for each symbol based on the provided transition matrix and emission probabilities."""
    num_states = transition_matrix.shape[0]
    
    # Create empty matrices
    trans_matrices = [np.zeros((num_states, num_states)) for _ in range(2)]
    
    # Populate the matrices
    for i in range(num_states):
        for j in range(2):  # 2 symbols: 0 and 1
            next_state = transition_matrix[i, j]
            trans_matrices[j][i, next_state] = emission_probabilities[i, j]

    return trans_matrices

def recurrent_state_transition_matrices(state_transition_matrix: np.ndarray, 
                                        emission_probabilities: np.ndarray, 
                                        recurrent_nodes: List[int]) -> np.ndarray:
    """Construct transition matrices for recurrent states of a subgraph."""
    num_states = len(recurrent_nodes)
    
    # Mapping of original state indices to recurrent indices
    state_mapping = {original: idx for idx, original in enumerate(recurrent_nodes)}
    
    # Create empty matrices for state transitions
    state_trans_matrices = [np.zeros((num_states, num_states)) for _ in range(2)]
    
    # Populate the matrices
    for original_idx in recurrent_nodes:
        for j in range(2):  # 2 symbols: 0 and 1
            next_state = state_transition_matrix[original_idx, j]
            if next_state in recurrent_nodes:
                i = state_mapping[original_idx]
                k = state_mapping[next_state]
                state_trans_matrices[j][i, k] = emission_probabilities[original_idx, j]

    return np.array(state_trans_matrices)

def create_random_epsilon_machine(num_states: int = 5, 
                                  num_symbols: int = 2, 
                                  alpha: float = 1.0) -> Tuple[nx.DiGraph, np.ndarray]:
    """Generate a random epsilon machine and return its recurrent subgraph and state transition matrices."""
    state_transition_matrix, emission_probabilities = create_markov_chain(num_states, num_symbols, alpha)
    G = transition_to_graph(state_transition_matrix, num_symbols)
    H = get_recurrent_subgraph(G)
    
    # Extract state transition matrices for the recurrent subgraph
    recurrent_trans_matrices = recurrent_state_transition_matrices(state_transition_matrix, emission_probabilities, H.nodes)

    return H, np.array(recurrent_trans_matrices)

def calculate_steady_state_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute the steady state of a given transition matrix."""

    # If 3D, sum along the first axis to get the combined transition matrix
    if transition_matrix.ndim == 3:
        transition_matrix = np.sum(transition_matrix, axis=0)
    
    # Define the identity matrix
    identity_matrix = np.eye(transition_matrix.shape[0])
    
    # Define the augmented matrix and right-hand side
    augmented_matrix = np.vstack([transition_matrix.T - identity_matrix, np.ones(transition_matrix.shape[0])])
    rhs = np.zeros(transition_matrix.shape[0])
    rhs = np.append(rhs, 1)

    # Solve for the steady state
    steady_state = LA.lstsq(augmented_matrix, rhs, rcond=None)[0]

    # make sure everything that is negative is really close to 0, then make it zero
    steady_state[np.abs(steady_state) < 1e-10] = 0

    
    # make sure steady state is a probability distribution
    steady_state = steady_state / np.sum(steady_state)
    
    return steady_state

def is_distribution_close(distribution: np.ndarray, known_distributions: List[np.ndarray], threshold: float) -> bool:
    """Check if the distribution is close to any of the known distributions."""
    for known_distribution in known_distributions:
        if np.linalg.norm(distribution - known_distribution) < threshold:
            return True
    return False

def compute_next_distribution(epsilon_machine: np.ndarray, X_current: np.ndarray, output: int) -> np.ndarray:
    """Compute the next mixed state distribution for a given output."""
    X_next = np.einsum('sd, s -> d', epsilon_machine[output], X_current)
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next


def to_mixed_state_presentation(epsilon_machine: np.ndarray, 
                                max_depth: int = 50, 
                                threshold: float = 1e-6) -> np.ndarray:
    """Convert an epsilon machine to its mixed state presentation."""
    n_outputs = epsilon_machine.shape[0]
    
    # Initialization
    tree = [{}]
    X = calculate_steady_state_distribution(epsilon_machine[0] + epsilon_machine[1])
    tree[0]['root'] = np.squeeze(X)
    seen_distributions = [X]
    state_index_map = {'root': 0}
    next_index = 1
    transition_matrices = np.zeros((n_outputs, max_depth, max_depth))
    
    # Tree exploration
    for depth in range(max_depth):
        tree.append({})
        all_branches_closed = True
        
        for node, X_current in tree[depth].items():
            for output in range(n_outputs):
                X_next = compute_next_distribution(epsilon_machine, X_current, output)
                new_node_name = f"{node}_{output}"
                
                if not is_distribution_close(X_next, seen_distributions, threshold):
                    all_branches_closed = False
                    tree[depth + 1][new_node_name] = X_next
                    seen_distributions.append(X_next)
                    state_index_map[new_node_name] = next_index
                    next_index += 1
                elif new_node_name not in state_index_map:
                    state_index_map[new_node_name] = state_index_map[node]
                    
                if next_index > transition_matrices.shape[1]:
                    new_size = 2 * transition_matrices.shape[1]
                    resized_matrices = np.zeros((n_outputs, new_size, new_size))
                    resized_matrices[:, :transition_matrices.shape[1], :transition_matrices.shape[1]] = transition_matrices
                    transition_matrices = resized_matrices
                    
                from_idx = state_index_map[node]
                to_idx = state_index_map[new_node_name]
                transition_prob = np.sum(X_current * np.einsum('sd, d -> s', epsilon_machine[output], X_next))
                transition_matrices[output, from_idx, to_idx] = transition_prob
        
        if all_branches_closed:
            break
    
    # Trim the dimensions
    return transition_matrices[:, :next_index, :next_index]

"""
def generate_emissions(epsilon_machine: np.ndarray, num_emissions: int) -> List[int]:
    #Generate a sequence of emissions from an epsilon machine
    # Get the number of outputs and states
    n_outputs, n_states, _ = epsilon_machine.shape
    
    # Calculate the steady-state distribution and use it to choose the initial state
    steady_state = calculate_steady_state_distribution(epsilon_machine[0] + epsilon_machine[1])
    steady_state = steady_state / np.sum(steady_state)
    current_state = np.random.choice(n_states, p=steady_state)
    
    emissions = []
    
    for _ in range(num_emissions):
        # Randomly choose an emission based on available outputs and transition probabilities
        prob0 = np.sum(epsilon_machine[0, current_state, :])
        prob1 = np.sum(epsilon_machine[1, current_state, :])
        emission_prob = np.array([prob0, prob1])
        emission_prob = emission_prob / np.sum(emission_prob)
        chosen_emission = np.random.choice(n_outputs, p=emission_prob)
        
        # Update the current state based on the chosen emission
        current_state = np.argmax(epsilon_machine[chosen_emission, current_state, :])
        # Record the emission
        emissions.append(chosen_emission)
    
    return emissions
"""