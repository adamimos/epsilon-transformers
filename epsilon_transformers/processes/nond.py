import numpy as np

from epsilon_transformers.processes.process import Presentation

class Nond(Presentation):
    """
    Class for generating the nond process, as defined in
    """

    def __init__(self):
        super().__init__()

    def _get_epsilon_machine(self, with_state_names=False):
        """
        Generate the epsilon machine for the nond process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((2, 3, 3))
        state_names = {'0': 0, '1': 1, '2': 2}
        T[0, 2, 0] = 1.0
        T[1, 0, 1] = 0.5
        T[1, 1, 1] = 0.5
        T[1, :, 2] = 1./3.

        if with_state_names:
            return T, state_names
        else:
            return T
        