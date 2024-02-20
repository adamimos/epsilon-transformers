import numpy as np
from typing import Dict

from epsilon_transformers.process.Process import Process

# TODO: Automatically generate PROCESS_REGISTRY using the inspect module
# TODO: Add test to make sure that all members of this module are a member of Process
# TODO: Find paper where mess3 process is introduced


class ZeroOneR(Process):
    def __init__(self, prob_of_zero_from_r_state: float = 0.5):
        self.p = prob_of_zero_from_r_state
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        state_names = {"0": 0, "1": 1, "R": 2}
        T[0, state_names["0"], state_names["1"]] = 1.0
        T[1, state_names["1"], state_names["R"]] = 1.0
        T[0, state_names["R"], state_names["0"]] = self.p
        T[1, state_names["R"], state_names["0"]] = 1 - self.p

        return T, state_names


class RRXOR(Process):
    def __init__(self, pR1=0.5, pR2=0.5):
        self.pR1 = pR1
        self.pR2 = pR2
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 5, 5))
        state_names = {"S": 0, "0": 1, "1": 2, "T": 3, "F": 4}
        T[0, state_names["S"], state_names["0"]] = self.pR1
        T[1, state_names["S"], state_names["1"]] = 1 - self.pR1
        T[0, state_names["0"], state_names["F"]] = self.pR2
        T[1, state_names["0"], state_names["T"]] = 1 - self.pR2
        T[0, state_names["1"], state_names["T"]] = self.pR2
        T[1, state_names["1"], state_names["F"]] = 1 - self.pR2
        T[1, state_names["T"], state_names["S"]] = 1.0
        T[0, state_names["F"], state_names["S"]] = 1.0

        return T, state_names


class Mess3(Process):
    def __init__(self, x=0.15, a=0.6):
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names


PROCESS_REGISTRY: Dict[str, Process] = {
    "z1r": ZeroOneR(),
    "rrxor": RRXOR(),
    "mess3": Mess3(),
}
