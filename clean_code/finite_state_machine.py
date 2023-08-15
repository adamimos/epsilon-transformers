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
    T: Dict[int, np.ndarray] = Field(default=None)

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
        T = np.zeros((2, num_states, num_states))

        # Fill the transition matrix and output matrix according to transition_function
        for key, value in self.transition_function.items():
            from_state = int(key[0])
            to_state = value
            output = int(key[1])
            
            if output == 0:
                self.transition_matrix[to_state, from_state] += self.emmision_0_probs[from_state]
                T[0, to_state, from_state] = self.emmision_0_probs[from_state]
            else:
                self.transition_matrix[to_state, from_state] += 1 - self.emmision_0_probs[from_state]
                T[1, to_state, from_state] = 1 - self.emmision_0_probs[from_state]
            
            self.transition_output_matrix[to_state, from_state] = output
            self.T = T
            # make sure that transition_matrix == T0 + T1
            assert np.allclose(self.transition_matrix, np.sum(T, axis=0))

        return self.transition_matrix, self.transition_output_matrix

