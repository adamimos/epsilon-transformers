import numpy as np

from epsilon_transformers.processes.process import Process


class Even(Process):
    """
    Class for generating EvenProcess data.
    """

    def __init__(self, p=2/3):
        self.p = p
        super().__init__()

    def _create_hmm(self):
        """
        Generate the epsilon machine for the EvenProcess.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 2, 2))
        state_names = {'E': 0, 'O': 1}
        T[1, state_names['E'], state_names['O']] = 1-self.p
        T[0, state_names['E'], state_names['E']] = self.p
        T[1, state_names['O'], state_names['E']] = 1.0


        if with_state_names:
            return T, state_names
        else:
            return T
        