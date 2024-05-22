"""

The HMM class represents a Hidden Markov Model
the system being modeled is assumed to be a Markov process with unobserved (hidden) states. 

The HMM class is defined by the following parameters:
- transition_probs: a matrix of transition probabilities of shape (emissions, states, states)
transition_probs[e, i, j] is the probability of transitioning from state i to j and emitting e
given that the model is in state i. The transition_probs matrix is
such that sum(transition_probs[e, i, j], axis=[0,2]) = 1 for each i

The class also calculates the stationary distribution of the model upon initialization.

This module also imports and uses the `calculate_steady_state_distribution` function from
the `markov_utilities` module to calculate the stationary distribution.

Classes:
    HMM: Represents a Hidden Markov Model.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class HMM:
    """
    A class representing a Hidden Markov Model.

    The HMM class is defined by the following parameters:
    - transition_probs: a 3D numpy array of shape (emissions, states, states)
    transition_probs[e, i, j] is the probability of transitioning from state i to j and emitting e
    such that sum(transition_probs[e, i, j], axis=[0,2]) = 1 for each i

    The class also calculates the stationary distribution of the model upon initialization.

    Attributes:
        transition_probs (np.ndarray): A 3D numpy array representing the transition probabilities.
        stationary_distribution (np.ndarray): A numpy array representing the stationary distribution
        calculated upon initialization.
    """

    transition_probs: np.ndarray
    stationary_distribution: np.ndarray = field(init=False)

    def __post_init__(self):
        self._validate_transition_probs()
        self.stationary_distribution = calculate_steady_state_distribution(
            self.transition_probs
        )

    def _validate_transition_probs(self):
        """Validates the shape and constraints of the transition probability matrix."""
        if self.transition_probs.ndim != 3:
            raise ValueError(
                "transition_probs must have shape (emissions, states, states)."
            )

        states_dim_match = (
            self.transition_probs.shape[1] == self.transition_probs.shape[2]
        )
        if not states_dim_match:
            raise ValueError(
                "transition_probs must be square in the last two dimensions."
            )

        sums_to_one = np.allclose(self.transition_probs.sum(axis=(0, 2)), 1)
        if not sums_to_one:
            raise ValueError(
                "Sum of probabilities over emissions and destination states must equal 1 for each origin state."
            )

    def __str__(self):
        return (
            f"HMM(transition_probs={self.transition_probs.shape}, "
            f"stationary_distribution={self.stationary_distribution})"
        )


def calculate_steady_state_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the steady state of a given transition matrix.

    Parameters:
    transition_matrix (np.ndarray): The transition matrix.

    Returns:
    np.ndarray: The steady state distribution.
    """
    if transition_matrix.ndim == 3:
        transition_matrix = np.sum(transition_matrix, axis=0)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)

    # Find the eigenvector corresponding to the eigenvalue 1
    steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real

    # Normalize the steady state vector
    steady_state_vector /= steady_state_vector.sum()

    # Select the first solution if there are multiple
    return steady_state_vector[:, 0]
