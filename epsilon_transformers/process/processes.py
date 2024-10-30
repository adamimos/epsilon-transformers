import numpy as np
from typing import cast
from types import FrameType
import inspect

from epsilon_transformers.process.Process import Process

# TODO: Write test for PROCESS_REGISTRY
# TODO: Think if you really need PROCESS_REGSITRY (if only getting called during dataloader creation, it may be better to have the dataloader take in a process)
# TODO: Add test to make sure that all members of this module are a member of Process
# TODO: Find paper where mess3 process is introduced
# TODO: Think through whether self.name is necessary (review it's usage in derive_mixed_state_presentation)
# TODO: Move _create_hmm into the init function prior to super()__init__()


class ZeroOneR(Process):
    def __init__(self, prob_of_zero_from_r_state: float = 0.5):
        self.name = "z1r"
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
        self.name = "rrxor"
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
        self.name = "mess3"
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

class Even(Process):
    def __init__(self):
        self.name = "Even"
        super().__init__()

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5   # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5   # From state 0, emit 1, go to state 1
        T[1,1,0] = 1.0   # From state 1, emit 1, go to state 0
        return T, state_names

class GoldenMean(Process):
    def __init__(self):
        self.name = "Golden"
        super().__init__()

    def _create_hmm(self):
        state_names = {"0": 0, "1": 1}
        T = np.zeros((2, 2, 2))
        T[0,0,0] = 0.5  # From state 0, emit 0, go to state 0
        T[1,0,1] = 0.5  # From state 0, emit 1, go to state 1
        T[0,1,0] = 1.0  # From state 1, emit 0, go to state 0 
        return T, state_names
 
class TransitionMatrixProcess(Process):
    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix
        super().__init__()

    def _create_hmm(self):
        return self.transition_matrix, {
            i: i for i in range(self.transition_matrix.shape[0])
        }


PROCESS_REGISTRY: dict[str, type] = {
    key: value
    # cast because we know the current frame has the above classes
    for key, value in cast(FrameType, inspect.currentframe()).f_locals.items()
    if isinstance(value, type) and issubclass(value, Process) and key != "Process"
}
