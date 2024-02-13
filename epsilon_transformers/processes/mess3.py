import numpy as np

from epsilon_transformers.processes.process import Process

# TODO: Find paper where mess3 process is introduced

class Mess3(Process):
    def __init__(self, x=0.15, a=0.6):
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
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
