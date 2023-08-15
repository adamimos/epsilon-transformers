import numpy as np
from numpy import linalg as LA
from typing import Tuple
from finite_state_machine import FiniteStateMachine
from complexity_measures import rate_distortion, calculate_steady_state_distribution

def generate_emission_0_probs(fsm: FiniteStateMachine, seed: int = None) -> np.ndarray:
    """
    Calculate the emission probabilities for each state in the FSM.
    """
    if fsm.emmision_0_probs is not None:
        return fsm.emmision_0_probs

    if seed is not None:
        np.random.seed(seed)

    # Initialize the emission probabilities as zeros
    fsm.emmision_0_probs = np.zeros(len(fsm.states))

    # Choose random probabilities for emissions
    for state in fsm.states:
        if str(state) + '0' in fsm.transition_function:
            if str(state) + '1' in fsm.transition_function:
                fsm.emmision_0_probs[state] = np.random.uniform()
            else:
                # If the state can only transition to 0, set the emission probability to 1
                fsm.emmision_0_probs[state] = 1
        else:
            # If the state cannot transition, set the emission probability to 0
            fsm.emmision_0_probs[state] = 0

    return fsm.emmision_0_probs
from numpy import linalg as LA


def from_hs(hs_line: str) -> FiniteStateMachine:
    # Remove the first 3 characters and the last character from the encoded FSM
    encoded_fsm = hs_line[3:-1]

    # Find the position of the first ']' character
    pos = encoded_fsm.find(']')

    # The number of states is given by the character before the first ']'
    num_states = int(encoded_fsm[pos-1])
    states = list(range(num_states))

    # The rest of the FSM encoding starts 3 characters after the first ']' and ends 'pos' characters from the end
    encoded_fsm = encoded_fsm[pos+3:-pos-3]

    # Initialize the transition function as an empty dictionary
    transition_function = {}

    # Parse the FSM encoding to fill in the transition function
    while len(encoded_fsm) >= 7:
        transition = encoded_fsm[:7]
        transition_function[str(int(transition[1])-1) + transition[3]] = int(transition[5]) - 1

        # Remove the transition we just processed from the FSM encoding
        if len(encoded_fsm) > 8:
            encoded_fsm = encoded_fsm[8:]
        else:
            encoded_fsm = []

    # Create a new instance of FiniteStateMachine, initialize emmission probabilities as zeros
    fsm = FiniteStateMachine(states=states, transition_function=transition_function)

    # Calculate emission probabilities
    fsm.emmision_0_probs = generate_emission_0_probs(fsm)

    # Calculate transition matrix
    fsm.calculate_transition_matrix()

    return fsm


def calculate_rate_distortion_curve(joint_prob_dist, betas = np.linspace(0, 100, 5000)):
    rate = []
    distortion = []
    for beta in betas:
        r, d = rate_distortion(joint_prob_dist, beta)
        rate.append(r)
        distortion.append(d)
    betas = np.hstack([betas, np.inf])
    # sort the rate and distortion by increasing rate
    rate, distortion = zip(*sorted(zip(rate, distortion)))
    return np.array(rate), np.array(distortion)




def create_history_sequences(binary_string, history_length):
    # Generate subsequences of binary_string of length history_length
    train = binary_string[:int(0.5*len(binary_string))]
    val = binary_string[int(0.5*len(binary_string)):int(0.7*len(binary_string))]
    test = binary_string[int(0.7*len(binary_string)):]

    train_history = np.asarray([train[i:i+history_length] for i in range(len(train) - history_length)])
    train_next = train[history_length:]

    val_history = np.asarray([val[i:i+history_length] for i in range(len(val) - history_length)])
    val_next = val[history_length:]

    test_history = np.asarray([test[i:i+history_length] for i in range(len(test) - history_length)])
    test_next = test[history_length:]
    return (train_history, train_next), (val_history, val_next), (test_history, test_next)


