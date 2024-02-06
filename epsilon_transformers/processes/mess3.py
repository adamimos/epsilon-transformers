import numpy as np

from epsilon_transformers.processes.process import Process

class Mess3(Process):
    """
    Class for generating the Mess3 process, as defined in
    """

    def __init__(self, x=0.15, a=0.6):
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
        """
        Generate the epsilon machine for the Mess3 process.

        Parameters:
        with_state_names (bool): If True, also return a dictionary mapping state names to indices.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices. Only returned if with_state_names is True.
        """
        T = np.zeros((3, 3, 3))
        state_names = {'A': 0, 'B': 1, 'C': 2}
        b = (1-self.a)/2
        y = 1-2*self.x

        ay = self.a*y
        bx = b*self.x
        by = b*y
        ax = self.a*self.x

        T[0, :, :] = [[ay, bx, bx],
                      [ax, by, bx],
                      [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx],
                      [bx, ay, bx],
                      [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax],
                      [bx, by, ax],
                      [bx, bx, ay]]

        return T, state_names
