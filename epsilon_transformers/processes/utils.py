import numpy as np
import networkx as nx
from typing import Tuple, List

NUM_STATES = 5
NUM_SYMBOLS = 2
ALPHA = 1.0

def random_markov_chain(num_states: int, num_symbols: int, alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create and return a Markov chain as transition matrix and emission probabilities."""
    # Transition matrix, T[i,j]=k means that when we go from state i and emit j, we go to state k
    transition_matrix = np.random.randint(num_states, size=(num_states, num_symbols))
    
    # Emission probabilities using a Dirichlet distribution
    # this creates a matrix of size (num_states, num_symbols) where each row sums to 1
    emission_probabilities = np.random.dirichlet([alpha] * num_symbols, size=num_states)

    return transition_matrix, emission_probabilities

def get_recurrent_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """Extract and return the largest strongly connected component from graph G."""
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(largest_scc).copy()
    return H

def transition_to_graph(transition_matrix: np.ndarray, num_symbols: int) -> nx.DiGraph:
    """Convert a transition matrix to a graph."""
    G = nx.DiGraph()
    for i in range(transition_matrix.shape[0]):
        for j in range(num_symbols):
            G.add_edge(i, transition_matrix[i, j], label=str(j))
    return G
    
def recurrent_state_transition_matrices(state_transition_matrix: np.ndarray, 
                                        emission_probabilities: np.ndarray, 
                                        recurrent_nodes: List[int]) -> np.ndarray:
    """Construct transition matrices for recurrent states of a subgraph."""
    num_states = len(recurrent_nodes)
    num_symbols = emission_probabilities.shape[1]
    
    # Mapping of original state indices to recurrent indices
    state_mapping = {original: idx for idx, original in enumerate(recurrent_nodes)}
    
    # Create empty matrices for state transitions
    state_trans_matrices = [np.zeros((num_states, num_states)) for _ in range(2)]
    
    # Populate the matrices
    for original_idx in recurrent_nodes:
        for j in range(num_symbols):  # usually 2 symbols: 0 and 1
            next_state = state_transition_matrix[original_idx, j]
            if next_state in recurrent_nodes:
                i = state_mapping[original_idx]
                k = state_mapping[next_state]
                state_trans_matrices[j][i, k] = emission_probabilities[original_idx, j]

    return np.array(state_trans_matrices)

def random(num_states: int = NUM_STATES, num_symbols: int = NUM_SYMBOLS, alpha: float = ALPHA) -> np.ndarray:
    """Generate a random epsilon machine and return its recurrent subgraph and state transition matrices."""
    state_transition_matrix, emission_probabilities = random_markov_chain(num_states, num_symbols, alpha)
    G = transition_to_graph(state_transition_matrix, num_symbols)
    H = get_recurrent_subgraph(G)
    
    # Extract state transition matrices for the recurrent subgraph
    recurrent_trans_matrices = recurrent_state_transition_matrices(state_transition_matrix, emission_probabilities, H.nodes)

    return recurrent_trans_matrices
