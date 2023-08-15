from typing import List, Dict
from pydantic import BaseModel, Field
import numpy as np
from numpy import linalg as LA
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class FiniteStateMachine(BaseModel):
    states: List[str] = Field(...)
    transition_function: Dict[str, str] = Field(...)
    emmision_0_probs: Dict[str, float] = Field(default=None)
    transition_matrix: np.ndarray = Field(default=None)
    transition_output_matrix: np.ndarray = Field(default=None)
    T: Dict[int, np.ndarray] = Field(default=None)
    state_indices: Dict[str, int] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.state_indices = {state: i for i, state in enumerate(self.states)}
        self.calculate_transition_matrix()


    def transition_exists(self, state: str, x: int) -> bool:
        return (state + str(x)) in self.transition_function

    def get_emission_prob(self, state: str, x: int) -> float:
        if x == 0:
            return self.emmision_0_probs[state]
        else:
            return 1 - self.emmision_0_probs[state]


    def simulate(self, time_steps: int, seed: int = None):
        if seed is not None:
            np.random.seed(seed)

        current_state = np.random.choice(self.states)
        past_states = [current_state]
        xs = []
        for _ in range(time_steps):
            x = np.random.choice([0, 1], p=[self.emmision_0_probs[current_state], 1 - self.emmision_0_probs[current_state]])
            xs.append(x)
            current_state = self.transition_function[current_state + str(x)]
            past_states.append(current_state)
        past_states = past_states[:-1]

        return xs, past_states

    def calculate_transition_matrix(self):

        if self.transition_matrix is not None:
            return self.transition_matrix, self.transition_output_matrix

        num_states = len(self.states)
        self.transition_matrix = np.zeros((num_states, num_states))
        self.transition_output_matrix = np.full((num_states, num_states), np.nan)
        T0 = np.zeros((num_states, num_states))
        T1 = np.zeros((num_states, num_states))

        for key, value in self.transition_function.items():
            from_state = self.state_indices[key[:-1]]
            from_state_name = key[:-1]
            to_state = self.state_indices[value]
            output = int(key[-1])

            if output == 0:
                self.transition_matrix[to_state, from_state] += self.emmision_0_probs[from_state_name]
                T0[to_state, from_state] = self.emmision_0_probs[from_state_name]
            else:
                self.transition_matrix[to_state, from_state] += 1 - self.emmision_0_probs[from_state_name]
                T1[to_state, from_state] = 1 - self.emmision_0_probs[from_state_name]
            
            self.transition_output_matrix[to_state, from_state] = output
            self.T = {0: T0, 1: T1}

        return self.transition_matrix, self.transition_output_matrix
