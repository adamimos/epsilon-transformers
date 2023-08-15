# Documentation for Finite State Machine Analysis

This project provides functionality to model a Finite State Machine (FSM), simulate its behavior, calculate various complexity measures, and visualize its state transitions and rate-distortion curve.

## Structure

The project is structured into four Python files:

1. `finite_state_machine.py`: Contains the definition of the `FiniteStateMachine` class and its methods. This is the core class in this project, representing a FSM.

2. `utils.py`: Contains utility functions that calculate emission probabilities, joint probability distribution, and rate-distortion curve for a FSM.

3. `complexity_measures.py`: Contains functions to calculate the entropy rate and statistical complexity of a FSM.

4. `visualization.py`: Contains functions to visualize a FSM and the rate-distortion curve.

5. `main.py`: Contains example usage of the above modules and functions.

## Modules and Functions

Here is an overview of the main modules and functions:

### `FiniteStateMachine`

This class represents a FSM. It includes methods for simulating the FSM and calculating its transition matrix. It takes as input a list of states, a transition function, and an array of emission probabilities for 0.

### `generate_emission_0_probs`

This function generates emission probabilities for each state in the FSM, given the FSM.

### `calculate_joint_prob_dist`

This function calculates the joint probability distribution for the FSM.

### `calculate_entropy_and_complexity`

This function calculates the entropy rate and statistical complexity of the FSM.

### `plot_from_transition_matrix`

This function takes a FSM as input and plots the FSM with nodes representing states and edges representing transitions.

### `calculate_rate_distortion_curve`

This function calculates the rate-distortion curve for a given joint probability distribution.

## Example Usage

See `main.py` for an example of how to define a FSM, calculate its entropy rate and statistical complexity, calculate the joint probability distribution and rate-distortion curve, and visualize the FSM and rate-distortion curve.

