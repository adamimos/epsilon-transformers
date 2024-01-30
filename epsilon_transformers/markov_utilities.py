import numpy as np
import networkx as nx
from numpy import linalg as LA
from typing import List, Tuple, Optional, Dict
from scipy.sparse import lil_matrix
from collections import defaultdict, deque
from collections import Counter
import numpy as np
from scipy.linalg import null_space

def generate_emissions(epsilon_machine: np.ndarray, num_emissions: int) -> List[int]:
    """
    Generate a sequence of emissions from an epsilon machine.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
                                  n_outputs is the number of possible outputs (emissions).
                                  n_states is the number of states in the machine.
                                  epsilon_machine[i, j, k] is the probability of transitioning from state j to state k and emitting output i.
    num_emissions (int): The number of emissions to generate.

    Returns:
    List[int]: The generated sequence of emissions.
    """

    # Get the number of outputs and states
    n_outputs, n_states, _ = epsilon_machine.shape
    
    # Calculate the steady-state distribution and use it to choose the initial state
    steady_state = calculate_steady_state_distribution(epsilon_machine.sum(axis=0))
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
    """
    Create and return a Markov chain as transition matrix and emission probabilities.

    Parameters:
    num_states (int): The number of states in the Markov chain.
    num_symbols (int): The number of possible symbols (emissions).
    alpha (float): The parameter for the Dirichlet distribution used to generate emission probabilities.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The transition matrix and emission probabilities.
    """
    
        # Transition matrix
    transition_matrix = np.random.randint(num_states, size=(num_states, num_symbols))
    
    # Emission probabilities using a Dirichlet distribution
    emission_probabilities = np.random.dirichlet([alpha] * num_symbols, size=num_states)
    
    return transition_matrix, emission_probabilities

def transition_to_graph(transition_matrix: np.ndarray, num_symbols: int) -> nx.DiGraph:
    """
    Convert a transition matrix to a graph.

    Parameters:
    transition_matrix (np.ndarray): The transition matrix of shape (num_states, num_symbols).
    num_symbols (int): The number of possible symbols (emissions).

    Returns:
    nx.DiGraph: The graph representation of the transition matrix.
    """
    
    G = nx.DiGraph()
    for i in range(transition_matrix.shape[0]):
        for j in range(num_symbols):
            G.add_edge(i, transition_matrix[i, j], label=str(j))
    return G


def get_recurrent_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Extract and return the largest strongly connected component from graph G.

    Parameters:
    G (nx.DiGraph): The input graph.

    Returns:
    nx.DiGraph: The largest strongly connected component of G.
    """
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(largest_scc).copy()
    return H

def construct_transition_matrices(transition_matrix: np.ndarray, emission_probabilities: np.ndarray) -> List[np.ndarray]:
    """
    Construct the transition matrices for each symbol based on the provided transition matrix and emission probabilities.

    Parameters:
    transition_matrix (np.ndarray): The transition matrix of shape (num_states, num_symbols).
    emission_probabilities (np.ndarray): The emission probabilities of shape (num_states, num_symbols).

    Returns:
    List[np.ndarray]: The list of transition matrices for each symbol.
    """    
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
    """
    Construct transition matrices for recurrent states of a subgraph.

    Parameters:
    state_transition_matrix (np.ndarray): The state transition matrix of shape (num_states, num_symbols).
    emission_probabilities (np.ndarray): The emission probabilities of shape (num_states, num_symbols).
    recurrent_nodes (List[int]): The list of recurrent nodes.

    Returns:
    np.ndarray: The transition matrices for recurrent states.
    """
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
    """
    Generate a random epsilon machine and return its recurrent subgraph and state transition matrices.

    Parameters:
    num_states (int): The number of states in the epsilon machine.
    num_symbols (int): The number of possible symbols (emissions).
    alpha (float): The parameter for the Dirichlet distribution used to generate emission probabilities.

    Returns:
    Tuple[nx.DiGraph, np.ndarray]: The recurrent subgraph and state transition matrices.
    """
    state_transition_matrix, emission_probabilities = create_markov_chain(num_states, num_symbols, alpha)
    G = transition_to_graph(state_transition_matrix, num_symbols)
    H = get_recurrent_subgraph(G)
    
    # Extract state transition matrices for the recurrent subgraph
    recurrent_trans_matrices = recurrent_state_transition_matrices(state_transition_matrix, emission_probabilities, H.nodes)

    return H, np.array(recurrent_trans_matrices)



def calculate_steady_state_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the steady state of a given transition matrix.

    Parameters:
    transition_matrix (np.ndarray): The transition matrix.

    Returns:
    np.ndarray: The steady state distribution.
    """
    if transition_matrix.ndim == 3:
        transition_matrix = np.sum(transition_matrix, axis=0)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find the eigenvector corresponding to the eigenvalue 1
    steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real


    # Normalize the steady state vector
    steady_state_vector /= steady_state_vector.sum()

    # Select the first solution if there are multiple
    return steady_state_vector[:, 0]


def is_distribution_close(distribution: np.ndarray, known_distributions: List[np.ndarray], threshold: float) -> bool:
    """
    Check if the distribution is close to any of the known distributions.

    Parameters:
    distribution (np.ndarray): The distribution to check.
    known_distributions (List[np.ndarray]): The list of known distributions.
    threshold (float): The threshold for the distance between distributions.

    Returns:
    bool: True if the distribution is close to any of the known distributions, False otherwise.
    """
    for known_distribution in known_distributions:
        if np.linalg.norm(distribution - known_distribution) < threshold:
            return True
    return False

def compute_next_distribution(epsilon_machine: np.ndarray, X_current: np.ndarray, output: int) -> np.ndarray:
    """
    Compute the next mixed state distribution for a given output.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
    X_current (np.ndarray): The current mixed state distribution.
    output (int): The output symbol.

    Returns:
    np.ndarray: The next mixed state distribution.
    """
    X_next = np.einsum('sd, s -> d', epsilon_machine[output], X_current)
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next


def to_mixed_state_presentation2(epsilon_machine: np.ndarray, 
                              max_depth: int = 50, 
                              threshold: float = 1e-6) -> np.ndarray:
   n_outputs = epsilon_machine.shape[0]
   tree = [{}]
   X = calculate_steady_state_distribution(epsilon_machine.sum(axis=0))
   tree[0]['root'] = np.squeeze(X)
   seen_distributions = {tuple(X)}
   state_index_map = {'root': 0}
   next_index = 1
   transition_matrices = np.zeros((n_outputs, max_depth, max_depth))

   for depth in range(max_depth):
       tree.append({})
       all_branches_closed = True

       for node, X_current in tree[depth].items():
           for output in range(n_outputs):
               X_next = compute_next_distribution(epsilon_machine, X_current, output)

               if np.sum(X_next) == 0.0:
                  continue

               new_node_name = f"{node}_{output}"

               if tuple(X_next) in seen_distributions:
                  to_idx = next(i for i, d in enumerate(seen_distributions) if np.all(np.linalg.norm(X_next - np.array(d)) < threshold))
               else:
                  all_branches_closed = False
                  tree[depth + 1][new_node_name] = X_next
                  seen_distributions.add(tuple(X_next))
                  state_index_map[new_node_name] = next_index
                  to_idx = next_index
                  next_index += 1

               if next_index > transition_matrices.shape[1]:
                  print("resizing")
                  print(tree)
                  new_size = 2 * transition_matrices.shape[1]
                  resized_matrices = np.zeros((n_outputs, new_size, new_size))
                  resized_matrices[:, :transition_matrices.shape[1], :transition_matrices.shape[1]] = transition_matrices
                  transition_matrices = resized_matrices

               from_idx = state_index_map[node]
               transition_prob = np.sum(np.einsum('s,sd->d', X_current, epsilon_machine[output]))

               transition_matrices[output, from_idx, to_idx] = transition_prob

       if all_branches_closed:
           break

   return transition_matrices[:, :next_index, :next_index]


def to_mixed_state_presentation(epsilon_machine: np.ndarray, 
                                max_depth: int = 50, 
                                threshold: float = 1e-6) -> np.ndarray:
    """
    Convert an epsilon machine to its mixed state presentation.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
    max_depth (int): The maximum depth for the tree exploration.
    threshold (float): The threshold for the distance between distributions.

    Returns:
    np.ndarray: The transition matrices for the mixed state presentation.
    """

    n_outputs = epsilon_machine.shape[0]
    
    # Initialization
    tree = [{}]
    X = calculate_steady_state_distribution(epsilon_machine.sum(axis=0))
    tree[0]['root'] = np.squeeze(X)
    seen_distributions = [tuple(X)]
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

                if np.sum(X_next) == 0.0:
                    continue

                new_node_name = f"{node}_{output}"
                
                # If the next mixed state is close to a previously seen state, add a transition back to the previously seen state
                if is_distribution_close(X_next, seen_distributions, threshold):
                    to_idx = seen_distributions.index(next(X for X in seen_distributions if np.all(np.linalg.norm(X_next - np.array(X)) < threshold)))
                else:
                    all_branches_closed = False
                    tree[depth + 1][new_node_name] = X_next
                    seen_distributions.append(tuple(X_next))
                    state_index_map[new_node_name] = next_index
                    to_idx = next_index
                    next_index += 1
                
                if next_index > transition_matrices.shape[1]:
                    print("resizing")
                    new_size = 2 * transition_matrices.shape[1]
                    resized_matrices = np.zeros((n_outputs, new_size, new_size))
                    resized_matrices[:, :transition_matrices.shape[1], :transition_matrices.shape[1]] = transition_matrices
                    transition_matrices = resized_matrices
                    
                from_idx = state_index_map[node]
                transition_prob = np.sum(np.einsum('s,sd->d', X_current, epsilon_machine[output]))
                
                transition_matrices[output, from_idx, to_idx] = transition_prob
        
        if all_branches_closed:
            break
    
    # Trim the dimensions
    return transition_matrices[:, :next_index, :next_index]



def to_mixed_state_presentation_sparse(epsilon_machine: np.ndarray, 
                                       max_depth: int = 50, 
                                       threshold: float = 1e-6) -> np.ndarray:
    """
    Convert an epsilon machine to its mixed state presentation.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
    max_depth (int): The maximum depth for the tree exploration.
    threshold (float): The threshold for the distance between distributions.

    Returns:
    np.ndarray: The transition matrices for the mixed state presentation.
    """
    n_outputs = epsilon_machine.shape[0]

    # make sure epsilon_machine has 3 sims
    assert epsilon_machine.ndim == 3

    
    # Initialization
    tree = [{}]
    X = calculate_steady_state_distribution(epsilon_machine.sum(axis=0))
    print(X)
    tree[0]['root'] = np.squeeze(X)
    seen_distributions = [tuple(X)]
    state_index_map = {'root': 0}
    next_index = 1
    transition_matrices = [lil_matrix((max_depth, max_depth)) for _ in range(n_outputs)]
    
    # Tree exploration
    for depth in range(max_depth):
        tree.append({})
        all_branches_closed = True
        
        for node, X_current in tree[depth].items():
            for output in range(n_outputs):
                X_next = compute_next_distribution(epsilon_machine, X_current, output)

                if np.sum(X_next) == 0.0:
                    continue

                new_node_name = f"{node}_{output}"
                
                # If the next mixed state is close to a previously seen state, add a transition back to the previously seen state
                if is_distribution_close(X_next, seen_distributions, threshold):
                    to_idx = seen_distributions.index(next(X for X in seen_distributions if np.all(np.linalg.norm(X_next - np.array(X)) < threshold)))
                else:
                    all_branches_closed = False
                    tree[depth + 1][new_node_name] = X_next
                    seen_distributions.append(tuple(X_next))
                    state_index_map[new_node_name] = next_index
                    to_idx = next_index
                    next_index += 1
                
                if next_index > transition_matrices[output].shape[1]:
                    new_size = 2 * transition_matrices[output].shape[1]
                    transition_matrices[output] = transition_matrices[output].resize((new_size, new_size))
                    
                from_idx = state_index_map[node]
                transition_prob = np.sum(np.einsum('s,sd->d', X_current, epsilon_machine[output]))
                
                transition_matrices[output][from_idx, to_idx] = transition_prob
        
        if all_branches_closed:
            break
    
    # Trim the dimensions
    return [matrix[:next_index, :next_index] for matrix in transition_matrices]

def to_probability_distributions(epsilon_machine: np.ndarray, 
                                 max_depth: int = 50, 
                                 threshold: float = 1e-6) -> List[np.ndarray]:
    """
    Convert an epsilon machine to a list of probability distributions over the states.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor.
    max_depth (int): The maximum depth for the tree exploration.
    threshold (float): The threshold for the distance between distributions.

    Returns:
    List[np.ndarray]: A list of probability distributions over the states.
    """

    n_outputs = epsilon_machine.shape[0]
    
    # Initialization
    tree = [{}]
    X = calculate_steady_state_distribution(epsilon_machine.sum(axis=0))
    distributions_list = [np.squeeze(X)]
    tree[0]['root'] = distributions_list[0]
    seen_distributions = [tuple(distributions_list[0])]
    state_index_map = {'root': 0}
    
    # Tree exploration
    for depth in range(max_depth):
        tree.append({})
        all_branches_closed = True
        
        for node, X_current in tree[depth].items():
            for output in range(n_outputs):
                X_next = compute_next_distribution(epsilon_machine, X_current, output)

                if np.sum(X_next) == 0.0:
                    continue

                new_node_name = f"{node}_{output}"

                # If the next mixed state is close to a previously seen state, skip adding it again
                if not is_distribution_close(X_next, seen_distributions, threshold):
                    all_branches_closed = False
                    tree[depth + 1][new_node_name] = X_next
                    seen_distributions.append(tuple(X_next))
                    distributions_list.append(X_next)
                
        if all_branches_closed:
            break

    return distributions_list


def epsilon_machine_to_graph(epsilon_machine: np.ndarray, state_names: Optional[Dict[str, int]] = None) -> nx.DiGraph:
    """
    Convert an epsilon machine to a graph.

    Parameters:
    epsilon_machine (np.ndarray): The epsilon machine transition tensor of shape (n_outputs, n_states, n_states).
                                  n_outputs is the number of possible outputs (emissions).
                                  n_states is the number of states in the machine.
                                  epsilon_machine[i, j, k] is the probability of transitioning from state j to state k given output i.
    state_names (Dict[str, int], optional): A dictionary mapping state names to state indices.

    Returns:
    nx.DiGraph: The graph representation of the epsilon machine.
    """
    # Get the number of outputs and states
    n_outputs, n_states, _ = epsilon_machine.shape

    # Create an empty directed graph
    G = nx.MultiDiGraph()

    # Invert the state_names dictionary if it's provided
    if state_names:
        state_names = {v: k for k, v in state_names.items()}

    # Add nodes to the graph
    for i in range(n_states):
        node_label = state_names[i] if state_names else i
        G.add_node(node_label)

    # Add edges to the graph for each transition in the epsilon machine
    for i in range(n_outputs):
        for j in range(n_states):
            for k in range(n_states):
                # Add an edge from state j to state k with label i and weight equal to the transition probability
                # only if the transition probability is not zero
                if epsilon_machine[i, j, k] != 0:
                    from_node = state_names[j] if state_names else j
                    to_node = state_names[k] if state_names else k
                    G.add_edge(from_node, to_node, label=str(i), weight=epsilon_machine[i, j, k])

    return G

def entropy(prob_dist, base=None):

    # 0*log(0) = 0
    prob_dist[prob_dist == 0] = 1

    return -np.sum(prob_dist * np.log(prob_dist))

def compute_myopic_entropy_from_MSP2(MSP_T, max_length=10):
  n_states = MSP_T.shape[1]

  # Vectorize the computation of emission_dist
  emission_dist = MSP_T.sum(axis=(1, 2))

  # Compute the entropy of the emissions from each mixed state
  mixed_state_entropy = entropy(emission_dist)

  # Compute the sum over the zeroth axis of the MSP_T tensor
  transition_matrix = MSP_T.sum(axis=0)

  # Reshape transition_matrix to ensure it has the correct shape
  transition_matrix = np.reshape(transition_matrix, (n_states, n_states))

  # Initialize the array for the myopic entropy
  hmu_L = np.empty(max_length)
  for L in range(max_length):
      # Compute the power of W_L
      transition_power = np.linalg.matrix_power(transition_matrix, L)
      # Compute the myopic entropy
      hmu = np.einsum('ij, j->i', transition_power, mixed_state_entropy)
      hmu_L[L] = hmu[0]

  return hmu_L



def compute_myopic_entropy_from_MSP(MSP_T, max_length=10):
    """
    Compute the myopic entropy from a Mixed State Presentation tensor.

    Parameters:
    MSP_T (np.ndarray): The Mixed State Presentation transition matrix tensor. (emissions, from_states, to_states)
    max_length (int, optional): The maximum length for the entropy computation. Defaults to 10.

    Returns:
    np.ndarray: The computed myopic entropy.
    """
    # Get the number of states
    n_states = MSP_T.shape[1]

    # Compute the entropy of the emissions from each mixed state
    mixed_state_entropy = np.zeros(n_states)
    for i in range(n_states):
        emission_dist = MSP_T[:, i, :].sum(axis=1)
        mixed_state_entropy[i] = entropy(emission_dist)

    # Compute the sum over the zeroth axis of the MSP_T tensor
    transition_matrix = MSP_T.sum(axis=0)

    # Initialize the list for the myopic entropy
    hmu_L = []
    for L in range(max_length):
        # Compute the power of W_L, considering that Python indexing starts at 0
        transition_power = np.linalg.matrix_power(transition_matrix, L)
        # Compute the myopic entropy
        hmu = np.einsum('ij, j->i', transition_power, mixed_state_entropy)
        hmu_L.append(hmu[0])

    return np.array(hmu_L)



def calculate_sequence_probabilities(matrix, max_length: int):
    """
    Calculate the probabilities of all possible sequences up to a given length.

    This function uses a breadth-first search to traverse the state space of the
    Markov chain defined by the given transition matrix. It calculates the probability
    of each sequence by multiplying the probabilities of the transitions that make up
    the sequence.

    Parameters:
    matrix (np.ndarray): The transition matrix of the Markov chain. The matrix should
        be a 3D array where the element at index (i, j, k) is the probability of
        transitioning from state j to state k with emission i.
    max_length (int): The maximum length of the sequences for which to calculate
        probabilities.

    Returns:
    dict: A dictionary where the keys are sequence lengths and the values are
        dictionaries mapping sequences of that length to their probabilities.
        e.g. {1: {'0': 0.5, '1': 0.5}, 2: {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}}
    """
    num_states = matrix.shape[1]
    num_emissions = matrix.shape[0]
    all_sequence_probs = defaultdict(lambda: defaultdict(float))
    queue = deque()

    steady_state_distribution = calculate_steady_state_distribution(matrix)

    for j in range(num_states):
        for i in range(num_emissions):
            for k in range(num_states):
                if matrix[i][j][k] > 0:
                    sequence = str(i)
                    transition_prob = matrix[i][j][k]
                    initial_prob = steady_state_distribution[j] * transition_prob
                    queue.append((sequence, k, initial_prob))
                    all_sequence_probs[1][sequence] += initial_prob

    while queue:
        sequence, curr_state, prob = queue.popleft()
        if len(sequence) < max_length:
            for i in range(num_emissions):
                for k in range(num_states):
                    if matrix[i][curr_state][k] > 0:
                        new_sequence = sequence + str(i)
                        transition_prob = matrix[i][curr_state][k]
                        new_prob = prob * transition_prob
                        queue.append((new_sequence, k, new_prob))
                        all_sequence_probs[len(new_sequence)][new_sequence] += new_prob

    return dict(all_sequence_probs)



def calculate_empirical_sequence_probabilities(sequence, max_length: int):
    sequence_length = len(sequence)
    all_sequence_probs = {}

    for length in range(1, max_length + 1):
        sequence_counts = Counter([sequence[i:i+length] for i in range(sequence_length - length + 1)])
        total_count = sum(sequence_counts.values())
        sequence_probs = {seq: count / total_count for seq, count in sequence_counts.items()}
        all_sequence_probs[length] = sequence_probs

    return all_sequence_probs
    

def create_transition_matrix(sequence_probs, N):
    # Get all length N sequences and their probabilities
    states = list(sequence_probs[N].keys())
    num_states = len(states)

    # Initialize the transition matrix and emission probabilities
    transition_matrix = np.zeros((2, num_states, num_states))
    emission_probs = np.zeros((2, num_states))

    # Create a mapping from sequences to state indices
    state_indices = {state: i for i, state in enumerate(states)}

    # Calculate the emission probabilities and transition probabilities
    for sequence, prob in sequence_probs[N+1].items():
        # The current state is the first N characters of the sequence
        current_state = sequence[:-1]
        # The next state is the last N characters of the sequence
        next_state = sequence[1:]
        # The emission is the last character of the sequence
        emission = int(sequence[-1])

        # Get the indices of the current state and next state
        current_state_index = state_indices[current_state]
        next_state_index = state_indices[next_state]

        # Update the transition matrix and emission probabilities
        transition_matrix[emission, current_state_index, next_state_index] += prob
        emission_probs[emission, current_state_index] += prob

    # Normalize the transition matrix and emission probabilities
    transition_matrix /= transition_matrix.sum(axis=(0, 2), keepdims=True)
    emission_probs /= emission_probs.sum(axis=0, keepdims=True)

    return states, transition_matrix, emission_probs

        