import numpy as np

from epsilon_transformers.processes.process import Presentation


class GoldenMean(Presentation):
    """
    Class for generating RKGoldenMean data.
    """
    def __init__(self, R, k, p):
        """
        Initialize the GoldenMeanProcess with R, k, p parameters.

        Parameters:
        R (int): The number of states that output 1.
        k (int): The number of states that output 0.
        p (float): The probability of outputting 1 in the final state.
        """
        self.R = R
        self.k = k
        self.p = p

        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the RKGoldenMean process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        assert self.k <= self.R, "k should be less than or equal to R"

        n_states = self.R + self.k
        T = np.zeros((2, n_states, n_states))

        # State names
        state_names = {chr(65 + i): i for i in range(n_states)}  # chr(65) is 'A'

        # First state
        T[1, state_names['A'], state_names['B']] = self.p
        T[0, state_names['A'], state_names['A']] = 1 - self.p

        # States that output 1
        for i in range(1, self.R):
            T[1, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # States that output 0
        for i in range(self.R, self.R+self.k-1):
            T[0, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # Last state
        T[0, state_names[chr(65 + n_states - 1)], state_names['A']] = 1.0

        if with_state_names:
            return T, state_names
        else:
            return T