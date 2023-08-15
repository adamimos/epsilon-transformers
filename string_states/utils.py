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

    # Choose random probabilities for emissions
    for state in fsm.states:
        if fsm.transition_exists(state, 0):
            if fsm.transition_exists(state, 1):
                fsm.set_emission_prob(state, 0, np.random.uniform())
            else:
                fsm.set_emission_prob(state, 0, 1)

    return fsm.emmision_0_probs


def create_history_sequences(binary_string, history_length):
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

def from_hs(hs_line: str) -> FiniteStateMachine:
    encoded_fsm = hs_line[3:-1]
    pos = encoded_fsm.find(']')
    num_states = int(encoded_fsm[pos-1])
    states = list(range(num_states))
    encoded_fsm = encoded_fsm[pos+3:-pos-3]

    fsm = FiniteStateMachine(states=states)

    while len(encoded_fsm) >= 7:
        transition = encoded_fsm[:7]
        fsm.add_transition(int(transition[1])-1, int(transition[3]), int(transition[5])-1)
        encoded_fsm = encoded_fsm[8:] if len(encoded_fsm) > 8 else []

    fsm.emmision_0_probs = generate_emission_0_probs(fsm)
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
    rate, distortion = zip(*sorted(zip(rate, distortion)))
    return np.array(rate), np.array(distortion)
