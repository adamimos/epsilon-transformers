import numpy as np

from epsilon_transformers.processes.process import Process


class ZeroOneR(Process):
    """
    Class for generating 01R data.
    """
    def __init__(self, prob_of_zero_from_r_state: float=0.5):
        self.p = prob_of_zero_from_r_state
        super().__init__()

    def _create_hmm(self):
        """
        Generate the epsilon machine for the 01R process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 3, 3))
        state_names = {'0': 0, '1': 1, 'R': 2}
        T[0, state_names['0'], state_names['1']] = 1.0
        T[1, state_names['1'], state_names['R']] = 1.0
        T[0, state_names['R'], state_names['0']] = self.p
        T[1, state_names['R'], state_names['0']] = 1-self.p

        return T, state_names