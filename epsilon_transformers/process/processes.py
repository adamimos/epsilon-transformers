import numpy as np
from typing import Dict

from epsilon_transformers.process.Process import Process

# TODO: Automatically generate PROCESS_REGISTRY using the inspect module
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

class EvenProcess(Process):
    def __init__(self, p: float = 0.5):
        self.name = "even"
        if not 0 <= p <= 1:
            raise ValueError("p must be a valid probability.")
        self.p = p
        super().__init__()

    def _create_hmm(self):
        # initialize the transition tensor
        T = np.zeros((2, 2, 2))
        state_names = {"E": 0, "O": 1}
        T[1, state_names["E"], state_names["O"]] = 1 - self.p
        T[0, state_names["E"], state_names["E"]] = self.p
        T[1, state_names["O"], state_names["E"]] = 1.0

        return T, state_names
    

class GoldenMean(Process):
    def __init__(self, R: int = 4, k: int = 1, p: float = 0.5):
        """
        Initialize the Golden Mean HMM.

        Parameters:
        R (int): The number of states that output 1.
        k (int): The number of states that output 0.
        p (float): The probability of outputting 1 in the final state.
        """
        assert k <= R, "k must be less than or equal to R"
        self.R = R
        self.k = k
        self.p = p
        self.name = "golden_mean"
        super().__init__()

    def _create_hmm(self):
        n_states = self.R + self.k
        state_names = {chr(65 + i): i for i in range(n_states)}  # chr(65) is 'A'

        T = np.zeros((2, n_states, n_states))

        # First state
        T[1, state_names["A"], state_names["B"]] = self.p
        T[0, state_names["A"], state_names["A"]] = 1 - self.p

        # States that output 1
        for i in range(1, self.R):
            T[1, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # States that output 0
        for i in range(self.R, self.R + self.k - 1):
            T[0, state_names[chr(65 + i)], state_names[chr(65 + i + 1)]] = 1.0

        # Last state
        T[0, state_names[chr(65 + n_states - 1)], state_names["A"]] = 1.0

        return T, state_names
    
class Nond(Process):
    def __init__(self):
        """
        Initialize the nond HMM as defined by Marzen and Crutchfield.
        Reference: https://arxiv.org/pdf/1702.08565.pdf
        """
        self.name = "nond"
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((2, 3, 3))
        T[0, 0, 2] = 1.0
        T[1, 1, 0] = 0.5
        T[1, 1, 1] = 0.5
        T[1, 2, :] = 1.0 / 3.0

        state_names = {"A": 0, "B": 1, "C": 2}
        return T, state_names

# TODO: This does not work! Figure out why!!
class Sierpinski(Process):
    def __init__(self, a: float = 1.0 / 6.0, s: float = 6.0):
        """
        Initialize the Sierpinski HMM as defined by Jurgens and Crutchfield.
        Reference: https://pubs.aip.org/aip/cha/article/31/8/083114/1059698
        Parameters:
        a (float): Parameter a in the model.
        s (float): Parameter s in the model.
        """
        self.a = a
        self.s = s
        self.name = "sierpinski"
        super().__init__()

    def _create_hmm(self):
        a_s = self.a * self.s
        one_minus_a_s = 1 - a_s
        one_minus_a_s_half = one_minus_a_s / 2
        a_s_minus_one_half_s = (one_minus_a_s * (self.s - 1)) / (2 * self.s)

        T = np.array(
            [
                [[self.a, 0, self.a * (self.s - 1)], [0, self.a, self.a * (self.s - 1)], [0, 0, a_s]],
                [
                    [one_minus_a_s_half, 0, 0],
                    [a_s_minus_one_half_s, one_minus_a_s_half, 0],
                    [a_s_minus_one_half_s, 0, one_minus_a_s_half],
                ],
                [
                    [one_minus_a_s_half, a_s_minus_one_half_s, 0],
                    [0, one_minus_a_s_half, 0],
                    [0, a_s_minus_one_half_s, a_s_minus_one_half_s],
                ],
            ]
        )

        state_names = {0: "A", 1: "B", 2: "C"}
        return T, state_names


PROCESS_REGISTRY: Dict[str, Process] = {
    "z1r": ZeroOneR(),
    "rrxor": RRXOR(),
    "mess3": Mess3(),
    "even": EvenProcess(),
    "golden_mean": GoldenMean(),
    "nond": Nond(),
    #"sierpinski": Sierpinski()
}


class TransitionMatrixProcess(Process):
    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix
        super().__init__()

    def _create_hmm(self):
        return self.transition_matrix, {i: i for i in range(self.transition_matrix.shape[0])}