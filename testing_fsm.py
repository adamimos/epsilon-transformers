#%%
%load_ext autoreload
%autoreload 2
from typing import List, Dict
from pydantic import BaseModel, Field
import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
# import mpatches
import matplotlib.patches as mpatches

# reload
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

    Methods
    -------
    simulate(time_steps: int, seed: int = None):
        Simulate the FSM for a given number of time steps.
    calculate_emmision_0_probs():
        Calculate the probabilities of emitting 0 for each state.
    calculate_joint_prob_dist_and_transition_matrix():
        Calculate the joint probability distribution and the transition matrix.
    calculate_entropy_and_complexity():
        Calculate the entropy rate and statistical complexity of the FSM.
    """

    states: List[int] = Field(...)
    transition_function: Dict[str, int] = Field(...)
    emmision_0_probs: np.ndarray = Field(default=None)
    transition_matrix: np.ndarray = Field(default=None)
    entropy_rate: float = Field(default=None)
    statistical_complexity: float = Field(default=None)
    transition_output_matrix: np.ndarray = Field(default=None)
    joint_prob_dist: np.ndarray = Field(default=None)

    @classmethod
    def from_hs(cls, hs_line: str):
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
        fsm = cls(states=states, transition_function=transition_function)

        # Calculate emission probabilities
        fsm.calculate_emmision_0_probs()

        # Calculate transition matrix
        fsm.calculate_transition_matrix()

        fsm.calculate_entropy_and_complexity()

        return fsm


    class Config:
        arbitrary_types_allowed = True

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



    def calculate_emmision_0_probs(self, seed: int = None):
        """
        Calculate the emission probabilities for each state in the FSM.
        """

        if seed is not None:
            np.random.seed(seed)

        # Initialize the emission probabilities as zeros
        self.emmision_0_probs = np.zeros(len(self.states))

        # Choose random probabilities for emissions
        for state in self.states:
            if str(state) + '0' in self.transition_function:
                if str(state) + '1' in self.transition_function:
                    self.emmision_0_probs[state] = np.random.uniform()
                else:
                    # If the state can only transition to 0, set the emission probability to 1
                    self.emmision_0_probs[state] = 1
            else:
                # If the state cannot transition, set the emission probability to 0
                self.emmision_0_probs[state] = 0

    def calculate_joint_prob_dis(self):
        num_states = len(self.states)
        joint_prob_dist = np.zeros((num_states, 2))

        eigenvalues, eigenvectors = LA.eig(self.transition_matrix)
        steady_state_indices = np.abs(eigenvalues - 1) < 1e-10
        steady_state_distribution = eigenvectors[:, steady_state_indices]
        steady_state_distribution /= np.sum(steady_state_distribution)

        for state in self.states:
            if str(state) + '0' in self.transition_function:
                joint_prob_dist[state, 0] = steady_state_distribution[state][0] * self.emmision_0_probs[state]
            if str(state) + '1' in self.transition_function:
                joint_prob_dist[state, 1] = steady_state_distribution[state][0] * (1 - self.emmision_0_probs[state])

        self.joint_prob_dist = joint_prob_dist

    def calculate_rate_distorion_curve(self, betas = np.linspace(0, 100, 5000)):
        rate = []
        distortion = []
        for beta in betas:
            r, d = rate_distortion(self.joint_prob_dist, beta)
            rate.append(r)
            distortion.append(d)
        betas = np.hstack([betas, np.inf])
        return np.array(rate), np.array(distortion)


    def calculate_entropy_and_complexity(self, recalculate: bool = False):
            if self.entropy_rate is not None and self.statistical_complexity is not None and not recalculate:
                return self.entropy_rate, self.statistical_complexity

            # Calculate the steady state distribution
            eigenvalues, eigenvectors = LA.eig(self.transition_matrix)
            steady_state_indices = np.abs(eigenvalues - 1) < 1e-10
            steady_state_distribution = eigenvectors[:, steady_state_indices]
            steady_state_distribution /= np.sum(steady_state_distribution)

            # Calculate p0 and p1
            p0 = 0
            accuracy = 0
            for i in range(len(self.states)):
                accuracy += steady_state_distribution[i] * np.max([self.emmision_0_probs[i], 1 - self.emmision_0_probs[i]])
                if self.emmision_0_probs[i] > 0.5:
                    p0 += steady_state_distribution[i]
            p1 = 1 - p0

            # Calculate entropy rate
            entropy = -p0 * np.log(p0) if p0 != 0 else 0
            entropy -= p1 * np.log(p1) if p1 != 0 else 0
            self.entropy_rate = -np.nansum(steady_state_distribution * np.log(steady_state_distribution))

            # Calculate statistical complexity
            complexities = np.zeros_like(self.emmision_0_probs)
            mask = self.emmision_0_probs != 0
            complexities[mask] = -self.emmision_0_probs[mask] * np.log(self.emmision_0_probs[mask])
            mask = self.emmision_0_probs != 1
            complexities[mask] -= (1 - self.emmision_0_probs[mask]) * np.log(1 - self.emmision_0_probs[mask])
            self.statistical_complexity = np.dot(complexities, steady_state_distribution)

            def deal_with_complex_part(value):
                if np.iscomplex(value):
                    complex_part = np.imag(value)
                    if complex_part < 1e-10:
                        return np.squeeze(np.real(value))
                    else:
                        raise ValueError("value is complex")
                else:
                    return np.squeeze(np.real(value))
                
            self.entropy_rate = deal_with_complex_part(self.entropy_rate)
            self.statistical_complexity = deal_with_complex_part(self.statistical_complexity)

                
            return self.entropy_rate, self.statistical_complexity
    
    def plot_from_transition_matrix(self):
        matrix = self.transition_matrix
        G = nx.DiGraph()

        # Add nodes
        for i in range(len(matrix)):
            G.add_node(i)

        # Add edges
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:  # If there's a transition from i to j
                    output = self.transition_output_matrix[i][j]
                    G.add_edge(j,i, weight=matrix[i][j], label=output)

        # options for the layout are 'spring', 'spectral', 'random', 'circular', 'shell', 'kamada_kawai'
        #pos = nx.kamada_kawai_layout(G, scale=0.5)
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

        # if entropy rate and statistical complexity have been calculated, add them in the bottom right
        if self.entropy_rate is not None and self.statistical_complexity is not None:
            # convert to string with 3 decimal places, if theres a complex number just disregard the complex part
            entropy_rate = str(np.round(self.entropy_rate, 3)).replace('j', 'i')
            statistical_complexity = str(np.round(self.statistical_complexity, 3)).replace('j', 'i')
            text_string = 'Entropy Rate = ' + entropy_rate + '\nStatistical Complexity = ' + statistical_complexity
            plt.text(0.5, -0.1, text_string, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.show()


    def compute_block_entropy(self, max_block_length: int, time_steps: int = 100000, seed: int = None):
        # Simulate the FSM
        outputs, _ = self.simulate(time_steps=time_steps, seed=seed)

        block_entropy = []
        for block_length in range(1, max_block_length + 1):
            # Count the occurrences of each block
            block_counts = {}
            for i in range(len(outputs) - block_length + 1):
                block = tuple(outputs[i:i+block_length])
                if block in block_counts:
                    block_counts[block] += 1
                else:
                    block_counts[block] = 1

            # Calculate the probabilities of each block
            total_blocks = sum(block_counts.values())
            block_probs = [count / total_blocks for count in block_counts.values()]

            # Compute the entropy
            entropy = -sum(p * np.log(p) for p in block_probs)
            block_entropy.append(entropy)

        return block_entropy


fsm = FiniteStateMachine.from_hs('Fa [1,2,3,4] [(1,0,2),(1,1,1),(2,0,3),(2,1,3),(3,0,4),(4,0,1)] [1,2,3,4]\n')
# outputs, states = fsm.simulate(time_steps=30000, seed=42)
fsm.calculate_entropy_and_complexity()
fsm.plot_from_transition_matrix()
# %%
# define the even process
# even process
transition_function = {'00':0, '11':0, '01':1}
emission_0_probs = np.array([2/3, 0])
fsm = FiniteStateMachine(states=[0,1], transition_function=transition_function,
                         emmision_0_probs=emission_0_probs)
fsm.calculate_transition_matrix()
fsm.calculate_entropy_and_complexity()
fsm.calculate_joint_prob_dis()
fsm.plot_from_transition_matrix()
# %%
sim_outputs, sim_states = fsm.simulate(30000)

# %%
r, d = fsm.calculate_rate_distorion_curve(betas=np.linspace(0,700,5000))
plt.plot(r, d)
plt.xlabel('Rate')
plt.ylabel('Accuracy')

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class GLM_predictor:

    def __init__(self, history_length=5):
        self.history_length = history_length
        self.model = None

    def _create_history_sequences(self, binary_string):
        # Generate subsequences of binary_string of length history_length
        train = binary_string[:int(0.5*len(binary_string))]
        val = binary_string[int(0.5*len(binary_string)):int(0.7*len(binary_string))]
        test = binary_string[int(0.7*len(binary_string)):]

        train_history = np.asarray([train[i:i+self.history_length] for i in range(len(train) - self.history_length)])
        train_next = train[self.history_length:]

        val_history = np.asarray([val[i:i+self.history_length] for i in range(len(val) - self.history_length)])
        val_next = val[self.history_length:]

        test_history = np.asarray([test[i:i+self.history_length] for i in range(len(test) - self.history_length)])
        test_next = test[self.history_length:]
        return (train_history, train_next), (val_history, val_next), (test_history, test_next)

    def train(self, binary_string):
        # Split binary_string into training and validation parts
        train_data, val_data, test_data = self._create_history_sequences(binary_string)
        train_history, train_next = train_data
        val_history, val_next = val_data
        test_history, test_next = test_data

        # Train models with L1 and L2 regularization
        # Train Logistic Regression models with L1 and L2 regularization
        l1_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100000)
        l1_model.fit(train_history, train_next)

        l2_model = LogisticRegression(penalty='l2', max_iter=100000)  # 'lbfgs' solver supports 'l2'
        l2_model.fit(train_history, train_next)

        # Evaluate models on validation set and select the best one
        score_l1 = l1_model.score(val_history, val_next)
        score_l2 = l2_model.score(val_history, val_next)
        self.model = l1_model if score_l1 > score_l2 else l2_model

        # do the prediction
        predicted_next_bits = self.model.predict(test_history)
        return predicted_next_bits, test_next




# %%
def empirical_relative_diversity_and_accuracy(actual_values, predicted_values):
    probability_of_one = np.mean(predicted_values == 1)
    probability_of_zero = 1 - probability_of_one

    if probability_of_one in {0, 1}:
        entropy = 0
    else:
        entropy = -probability_of_one * np.log(probability_of_one) - probability_of_zero * np.log(probability_of_zero)

    accuracy = np.mean((actual_values == 1) == (predicted_values == 1))

    return entropy, accuracy

# %%

r, d = fsm.calculate_rate_distorion_curve(betas=np.linspace(0,700,5000))
sim_outputs, sim_states = fsm.simulate(300000)
e, a = [], []
for history_length in range(1, 15):
    predictor = GLM_predictor(history_length=history_length)
    predicted_next_bits, actual_next_bits = predictor.train(sim_outputs)
    entropy, accuracy = empirical_relative_diversity_and_accuracy(actual_next_bits, predicted_next_bits)
    e.append(entropy)
    a.append(accuracy)


plt.plot(r, d)
for i in range(len(e)):
    plt.plot(e[i], a[i], '.r')
plt.xlabel('Rate')
plt.ylabel('Accuracy')

# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, d_head):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)
        self.W_O = nn.Linear(d_head, d_head, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(input_size, input_size)))

    def forward(self, x):
        # x is of size (batch_size, input_size, d_model)
        # get the query, key, and value
        Q = self.W_Q(x) # (batch_size, input_size, d_head)
        K = self.W_K(x) # (batch_size, input_size, d_head)
        V = self.W_V(x) # (batch_size, input_size, d_head)
        # get the attention weights
        A = torch.einsum("bid,bjd->bij", Q, K) / (d_head**0.5) 
        A = A.masked_fill(self.mask==0, float("-inf"))
        A = F.softmax(A, dim=-1) # the rows of A sum to 1
        # apply the attention weights
        O = torch.einsum("bij,bjd->bid", A, V) # this is the output of the attention head, we weight the values by the attention weights
        O = self.W_O(O) # (batch_size, input_size, d_model)
        return O 
    
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)
        
    def forward(self, x):
        # x is of size (batch_size, input_size, d_model)
        x = self.W_in(x) # (batch_size, input_size, d_mlp)
        x = F.relu(x)
        x = self.W_out(x) # (batch_size, input_size, d_model)
        return x

class Model(nn.Module):
    def __init__(self, d_vocab=2, d_model=16, input_size=3, d_head=4, d_mlp=4*16):
        super().__init__()
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(input_size, d_model)
        self.heads = nn.ModuleList([Head(d_head) for _ in range(4)])
        self.mlp = MLP(d_model, d_mlp)
        self.unembedding = nn.Linear(d_model, d_vocab)

    def forward(self, x):
        # x is of size (batch_size, input_size)
        # embed the input
        x = self.embedding(x) + self.pos_embedding(torch.arange(input_size, device=x.device))
        # apply the attention heads, stack them
        x = x + torch.cat([head(x) for head in self.heads], dim=-1) # (batch_size, input_size, d_model)
        # apply the MLP
        x = x + self.mlp(x)
        # unembed the output
        x = self.unembedding(x)
        return x
# %%
import numpy as np

def create_history_sequences(binary_string, history_length):
    # Generate subsequences of binary_string of length history_length
    train = binary_string[:int(0.5*len(binary_string))]
    val = binary_string[int(0.5*len(binary_string)):int(0.7*len(binary_string))]
    test = binary_string[int(0.7*len(binary_string)):]

    train_history = np.asarray([train[i:i+history_length] for i in range(len(train) - history_length)])
    train_next = train[history_length:]

    val_history = np.asarray([val[i:i+history_length] for i in range(len(val) - history_length)])
    val_next = val[history_length:]

    test_history = np.asarray([test[i:i+history_length] for i in range(len(test) - history_length)])
    test_next = test[history_length:]
    return (train_history, train_next), (val_history, val_next), (test_history, test_next)
# %%
import pandas as pd
import itertools
n_hist = 6
train, val, test = create_history_sequences(sim_outputs, n_hist)
# let's create a conditional historgram of train[1] conditioned on train[0]
# train[0] is a list of 3 bit sequences, so let's convert those to strings
train_0_strings = np.asarray(["".join([str(b) for b in seq]) for seq in train[0]])

all_possible_strings = np.asarray(["".join([str(b) for b in seq]) for seq in np.asarray(list(itertools.product([0,1], repeat=n_hist)))])

next_bits = np.asarray(train[1])
num_strings = len(all_possible_strings)
prob_01_given_string = np.zeros((num_strings, 2))
for i,s in enumerate(all_possible_strings):
    next_bit = next_bits[train_0_strings==s]
    total = len(next_bit)
    print(s, np.sum(next_bit==0)/total, np.sum(next_bit==1)/total)
    prob_01_given_string[i,0] = np.sum(next_bit==0)/total
    prob_01_given_string[i,1] = np.sum(next_bit==1)/total


plt.figure(figsize=(10,10))
plt.matshow(prob_01_given_string.T, cmap="viridis", fignum=1)
# xlabels are the strings
plt.xticks(np.arange(num_strings), all_possible_strings)
plt.title("Conditional probability of next bit given previous 3 bits")

#%%
# lets sort the data by the probability of the next bit being 1
sorted_idx = np.argsort(prob_01_given_string[:,1])
plt.figure(figsize=(10,10))
plt.matshow(prob_01_given_string[sorted_idx].T, cmap="Reds", fignum=1)
# xlabels are the strings, put them on the bottom
plt.xticks(np.arange(num_strings), all_possible_strings[sorted_idx], rotation=45, ha="left")
plt.title("Conditional probability of next bit given previous 3 bits")
# add vertical lines to seperate each string
for i in range(num_strings-1):
    plt.axvline(i+0.5, color="k")

# put the probabiliteis in white text in each cell, make sure to use the sorted index
# the higher the probability, the whiter the text should be, gradient from black to white
for i in range(num_strings):
    for j in range(2):
        prob = prob_01_given_string[sorted_idx][i,j]*100
        color = "k" if prob < 0.5 else "w"
        plt.text(i, j, f"{prob:.0f}%", color=color, ha="center", va="center")

# if value is nan, then make the cell checkerboard pattern
for i in range(num_strings):
    for j in range(2):
        if np.isnan(prob_01_given_string[sorted_idx][i,j]):
            plt.text(i, j, "N/A", color="k", ha="center", va="center")
            # make checkerboard pattern

# add colorbar, horizontal
plt.colorbar(orientation="horizontal", pad=0.1, aspect=50).set_label("probability")
# add colorbar title "probability"
plt.xlabel("history")
plt.ylabel("next bit")
plt.show()

# %%
# make the previous plot in bokeh


# %%
# fsm.py

