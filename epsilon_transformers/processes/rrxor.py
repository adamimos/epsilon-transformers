import numpy as np
import random

from epsilon_transformers.processes.process import Process

class RRXOR(Process):

    def __init__(self, pR1=0.5, pR2=0.5):
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _create_hmm(self):
        """
        Generate the epsilon machine for the RRXOR process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping
                                state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if
              with_state_names is True.
        """
        T = np.zeros((2, 5, 5))
        state_names = {'S': 0, '0': 1, '1': 2, 'T': 3, 'F': 4}
        T[0, state_names['S'], state_names['0']] = self.pR1
        T[1, state_names['S'], state_names['1']] = 1 - self.pR1
        T[0, state_names['0'], state_names['F']] = self.pR2
        T[1, state_names['0'], state_names['T']] = 1 - self.pR2
        T[0, state_names['1'], state_names['T']] = self.pR2
        T[1, state_names['1'], state_names['F']] = 1 - self.pR2
        T[1, state_names['T'], state_names['S']] = 1.0
        T[0, state_names['F'], state_names['S']] = 1.0

        return T, state_names

