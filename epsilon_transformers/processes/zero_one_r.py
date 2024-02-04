import numpy as np

from epsilon_transformers.processes.process import Presentation


class ZeroOneR(Presentation):
    """
    Class for generating 01R data.
    """

    def __init__(self, p=0.5):
        self.p = p # probability of emitting 0 from the R state
        super().__init__()

    def _get_epsilon_machine(self, with_state_names:bool=False):
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

        if with_state_names:
            return T, state_names
        else:
            return T