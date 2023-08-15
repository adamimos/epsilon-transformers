# %%

%load_ext autoreload
%autoreload 2
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
# import mpatches
import matplotlib.patches as mpatches

class FiniteStateMachine(BaseModel):
    """
    A class to represent a Finite State Machine (FSM).

    Attributes
    ----------
    states : List[int]
        The states of the FSM.
    transition_function : Dict[str, int]
        The transition function of the FSM, represented as a dictionary mapping from
        state-action pairs to next states.
    emmision_0_probs : np.ndarray
        The probabilities of emitting 0 from each state.
    """

    states: List[int] = Field(...)
    transition_function: Dict[str, int] = Field(...)
    emmision_0_probs: np.ndarray = Field(default=None)
    transition_matrix: np.ndarray = Field(default=None)
    transition_output_matrix: np.ndarray = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.emmision_0_probs is None:
            self.emmision_0_probs = generate_emission_0_probs(self)
        self.calculate_transition_matrix()

    def simulate(self, time_steps: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        # Run the simulation
        current_state = np.random.randint(len(self.states))
        past_states = [current_state]
        xs = []
        for _ in range(time_steps):
            x = np.random.choice([0, 1], p=[self.emmision_0_probs[current_state], 1 - self.emmision_0_probs[current_state]])
            xs.append(x)
            current_state = self.transition_function[str(current_state) + str(x)]
            past_states.append(current_state)
        past_states = past_states[:-1]

        return xs, past_states

    def calculate_transition_matrix(self):
        """
        Calculate the transition matrix for the FSM.
        """

        # check that we have all the necessary information in order to calculate the transition matrix
        if self.transition_matrix is not None:
            return self.transition_matrix, self.transition_output_matrix
        
        # we need emission probabilities in order to calculate the transition matrix
        if self.emmision_0_probs is None:
            self.emmision_0_probs = generate_emission_0_probs(self)

        num_states = len(self.states)
        # Initialize the transition matrix and output matrix as zeros
        self.transition_matrix = np.zeros((num_states, num_states))
        self.transition_output_matrix = np.full((num_states, num_states), np.nan)

        # Fill the transition matrix and output matrix according to transition_function
        for key, value in self.transition_function.items():
            from_state = int(key[0])
            to_state = value
            output = int(key[1])
            
            if key[1] == '0':
                self.transition_matrix[to_state, from_state] += self.emmision_0_probs[from_state]
            else:
                self.transition_matrix[to_state, from_state] += 1 - self.emmision_0_probs[from_state]
            
            self.transition_output_matrix[to_state, from_state] = output

        return self.transition_matrix, self.transition_output_matrix


# %%
def generate_emission_0_probs(fsm: FiniteStateMachine, seed: int = None) -> np.ndarray:
    """
    Calculate the emission probabilities for each state in the FSM.
    """
    if fsm.emmision_0_probs is not None:
        return fsm.emmision_0_probs

    if seed is not None:
        np.random.seed(seed)

    # Initialize the emission probabilities as zeros
    fsm.emmision_0_probs = np.zeros(len(fsm.states))

    # Choose random probabilities for emissions
    for state in fsm.states:
        if str(state) + '0' in fsm.transition_function:
            if str(state) + '1' in fsm.transition_function:
                fsm.emmision_0_probs[state] = np.random.uniform()
            else:
                # If the state can only transition to 0, set the emission probability to 1
                fsm.emmision_0_probs[state] = 1
        else:
            # If the state cannot transition, set the emission probability to 0
            fsm.emmision_0_probs[state] = 0

    return fsm.emmision_0_probs
from numpy import linalg as LA

def calculate_joint_prob_dist(fsm: FiniteStateMachine) -> np.ndarray:
    num_states = len(fsm.states)
    joint_prob_dist = np.zeros((num_states, 2))

    eigenvalues, eigenvectors = LA.eig(fsm.transition_matrix)
    steady_state_indices = np.abs(eigenvalues - 1) < 1e-10
    steady_state_distribution = eigenvectors[:, steady_state_indices]
    steady_state_distribution /= np.sum(steady_state_distribution)

    for state in fsm.states:
        if str(state) + '0' in fsm.transition_function:
            joint_prob_dist[state, 0] = steady_state_distribution[state][0] * fsm.emmision_0_probs[state]
        if str(state) + '1' in fsm.transition_function:
            joint_prob_dist[state, 1] = steady_state_distribution[state][0] * (1 - fsm.emmision_0_probs[state])

    return joint_prob_dist



def calculate_entropy_and_complexity(fsm: FiniteStateMachine, recalculate: bool = False) -> Tuple[float, float]:
    # Calculate the steady state distribution
    eigenvalues, eigenvectors = LA.eig(fsm.transition_matrix)
    steady_state_indices = np.abs(eigenvalues - 1) < 1e-10
    steady_state_distribution = eigenvectors[:, steady_state_indices]
    steady_state_distribution /= np.sum(steady_state_distribution)

    # Calculate p0 and p1
    p0 = 0
    accuracy = 0
    for i in range(len(fsm.states)):
        accuracy += steady_state_distribution[i] * np.max([fsm.emmision_0_probs[i], 1 - fsm.emmision_0_probs[i]])
        if fsm.emmision_0_probs[i] > 0.5:
            p0 += steady_state_distribution[i]
    p1 = 1 - p0

    # Calculate entropy rate
    entropy = -p0 * np.log(p0) if p0 != 0 else 0
    entropy -= p1 * np.log(p1) if p1 != 0 else 0
    entropy_rate = -np.nansum(steady_state_distribution * np.log(steady_state_distribution))

    # Calculate statistical complexity
    complexities = np.zeros_like(fsm.emmision_0_probs)
    mask = fsm.emmision_0_probs != 0
    complexities[mask] = -fsm.emmision_0_probs[mask] * np.log(fsm.emmision_0_probs[mask])
    mask = fsm.emmision_0_probs != 1
    complexities[mask] -= (1 - fsm.emmision_0_probs[mask]) * np.log(1 - fsm.emmision_0_probs[mask])
    statistical_complexity = np.dot(complexities, steady_state_distribution)

    def deal_with_complex_part(value):
        if np.iscomplex(value):
            complex_part = np.imag(value)
            if complex_part < 1e-10:
                return np.squeeze(np.real(value))
            else:
                raise ValueError("value is complex")
        else:
            return np.squeeze(np.real(value))
                
    entropy_rate = deal_with_complex_part(entropy_rate)
    statistical_complexity = deal_with_complex_part(statistical_complexity)

    return entropy_rate, statistical_complexity


import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_from_transition_matrix(fsm: FiniteStateMachine):
    matrix = fsm.transition_matrix
    G = nx.DiGraph()

    # Add nodes
    for i in range(len(matrix)):
        G.add_node(i)

    # Add edges
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:  # If there's a transition from i to j
                output = fsm.transition_output_matrix[i][j]
                G.add_edge(j,i, weight=matrix[i][j], label=output)

    pos = nx.circular_layout(G)

    # Get edge weights and normalize for drawing
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # edge color is blue if output is 0, red if output is 1
    edge_colors = ['b' if G[u][v]['label'] == 0 else 'r' for u, v in G.edges()]

    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=500)
    nx.draw_networkx_labels(G, pos, font_color='white')

    # Plot edges with alpha values proportional to the weights
    for i in range(len(weights)):
        edge = list(G.edges())[i]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_colors[i],
                               width=2, alpha=weights[i], connectionstyle='arc3, rad=0.1')

    # add a key for the edge label colors
    red_patch = mpatches.Patch(color='red', label='Output = 1')
    blue_patch = mpatches.Patch(color='blue', label='Output = 0')
    plt.legend(handles=[red_patch, blue_patch])

    plt.show()
    

# The rate_distortion function used in calculate_rate_distortion_curve is not defined in the provided code.
# Here's a placeholder implementation that just returns random values.
# You'll need to replace this with your actual rate_distortion function.
def rate_distortion(joint_prob_dist, beta):
    return np.random.random(), np.random.random()


def from_hs(hs_line: str) -> FiniteStateMachine:
    # Remove the first 3 characters and the last character from the encoded FSM
    encoded_fsm = hs_line[3:-1]

    # Find the position of the first ']' character
    pos = encoded_fsm.find(']')

    # The number of states is given by the character before the first ']'
    num_states = int(encoded_fsm[pos-1])
    states = list(range(num_states))

    # The rest of the FSM encoding starts 3 characters after the first ']' and ends 'pos' characters from the end
    encoded_fsm = encoded_fsm[pos+3:-pos-3]

    # Initialize the transition function as an empty dictionary
    transition_function = {}

    # Parse the FSM encoding to fill in the transition function
    while len(encoded_fsm) >= 7:
        transition = encoded_fsm[:7]
        transition_function[str(int(transition[1])-1) + transition[3]] = int(transition[5]) - 1

        # Remove the transition we just processed from the FSM encoding
        if len(encoded_fsm) > 8:
            encoded_fsm = encoded_fsm[8:]
        else:
            encoded_fsm = []

    # Create a new instance of FiniteStateMachine, initialize emmission probabilities as zeros
    fsm = FiniteStateMachine(states=states, transition_function=transition_function)

    # Calculate emission probabilities
    fsm.emmision_0_probs = generate_emission_0_probs(fsm)

    # Calculate transition matrix
    fsm.calculate_transition_matrix()

    return fsm


# %%
def rate_distortion(p, beta, tol=1e-6, max_iter=5000):
    """
    Calculate the rate-distortion function for a given joint probability distribution p and a beta value.

    Args:
    p: Joint probability distribution.
    beta: Beta value, which can be interpreted as the inverse temperature in statistical physics.
    tol: Tolerance for convergence. If the change in marginal_pXhat and conditional_pXhat_given_S is less than tol,
         the function will stop iterating and return the current rate and distortion.
    max_iter: Maximum number of iterations.

    Returns:
    R: The rate value in the rate-distortion function.
    D: The distortion value in the rate-distortion function.
    """
    
    # Calculate the marginal probabilities of X and S
    marginal_pX = np.sum(p, 0)
    marginal_pS = np.sum(p, 1)

    # Calculate conditional probability of X given S
    conditional_pX_given_S = np.dot(np.diag(1 / marginal_pS), p)

    # Distortion matrix is equal to the conditional probability of X given S
    distortion_matrix = conditional_pX_given_S

    # Initialize the conditional probability of Xhat given S randomly
    initial_conditional_pXhat_given_S0 = np.random.uniform(size=len(marginal_pS))
    conditional_pXhat_given_S = np.vstack([initial_conditional_pXhat_given_S0, 1 - initial_conditional_pXhat_given_S0]).T

    # Calculate the marginal probability of Xhat
    marginal_pXhat = np.dot(conditional_pXhat_given_S.T, marginal_pS)

    # Iterate to refine the conditional probability of Xhat given S and marginal probability of Xhat
    for _ in range(max_iter):
        # Calculate new estimates
        log_conditional_pXhat_given_S = np.meshgrid(np.log(marginal_pXhat), np.ones(len(marginal_pS)))[0] + beta * distortion_matrix
        new_conditional_pXhat_given_S = np.exp(log_conditional_pXhat_given_S)
        normalization_constants = np.sum(new_conditional_pXhat_given_S, 1)
        new_conditional_pXhat_given_S = np.dot(np.diag(1 / normalization_constants), new_conditional_pXhat_given_S)
        new_marginal_pXhat = np.dot(new_conditional_pXhat_given_S.T, marginal_pS)
        
        # Check for convergence
        if np.allclose(marginal_pXhat, new_marginal_pXhat, atol=tol) and np.allclose(conditional_pXhat_given_S, new_conditional_pXhat_given_S, atol=tol):
            # print('beta=', beta, 'converged after', _+1, 'iterations')
            break
        
        # Update estimates
        marginal_pXhat = new_marginal_pXhat
        conditional_pXhat_given_S = new_conditional_pXhat_given_S

    # Calculate the rate value R in the rate-distortion function
    R = -np.nansum(marginal_pXhat * np.log(marginal_pXhat)) + np.dot(marginal_pS, np.nansum(conditional_pXhat_given_S * np.log(conditional_pXhat_given_S), 1))

    # Calculate the distortion value D in the rate-distortion function
    D = np.dot(marginal_pS, np.sum(conditional_pXhat_given_S * conditional_pX_given_S, 1))

    return R, D

# %%
def calculate_rate_distortion_curve(joint_prob_dist, betas = np.linspace(0, 100, 5000)):
    rate = []
    distortion = []
    for beta in betas:
        r, d = rate_distortion(joint_prob_dist, beta)
        rate.append(r)
        distortion.append(d)
    betas = np.hstack([betas, np.inf])
    return np.array(rate), np.array(distortion)

# %%

# Define the FSM, this is the even process
transition_function = {'00':0, '11':0, '01':1}
emission_0_probs = np.array([2/3, 0])
fsm = FiniteStateMachine(states=[0,1], transition_function=transition_function,
                         emmision_0_probs=emission_0_probs)

# Calculate entropy rate and statistical complexity
entropy_rate, statistical_complexity = calculate_entropy_and_complexity(fsm)
print(f"Entropy Rate: {entropy_rate}")
print(f"Statistical Complexity: {statistical_complexity}")

# Calculate joint probability distribution and rate distortion curve
joint_prob_dist = calculate_joint_prob_dist(fsm)
print(f"Joint Probability Distribution:\n {joint_prob_dist}")
betas = np.linspace(0, 100, 5000)
rate, distortion = calculate_rate_distortion_curve(joint_prob_dist, betas)
print(f"Rate: {rate}")
print(f"Distortion: {distortion}")

# plot the FSM and rate distortion curve
plot_from_transition_matrix(fsm)
plt.plot(rate, distortion)
plt.xlabel('Rate')
plt.ylabel('Accuracy')


# %%
