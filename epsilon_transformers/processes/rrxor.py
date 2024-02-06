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

        if with_state_names:
            return T, state_names
        else:
            return T

    def generate_without_epsilon_machine(self, total_length: int, with_positions=False):
        """
        Generate a sequence of Random-Random-XOR (RRXOR) data.

        Parameters:
        total_length (int): The total length of the sequence to generate.
        with_positions (bool): If True, also return a list of positions ("R1", "R2", "XOR").

        Returns:
        list: The generated RRXOR sequence. If with_positions is True, also return a list of positions.
        """
        output = []
        positions = []
        
        while len(output) < total_length + 3:
            bit1 = random.randint(0, 1)
            bit2 = random.randint(0, 1)
            xor_result = bit1 ^ bit2
            output.extend([str(bit1), str(bit2), str(xor_result)])
            positions.extend(["R1", "R2", "XOR"])
        
        # Start the sequence randomly at bit 1,2 r 3
        start_index = random.randint(0, 2)
        output = output[start_index:]
        positions = positions[start_index:]

        # Return the sequence up to the desired total length along with positions
        if with_positions:
            return output[:total_length], positions[:total_length]
        else:
            return output[:total_length]

