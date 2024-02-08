from typing import List, Tuple, Optional
import numpy as np


class Mixed_State_Tree:
    def __init__(
        self,
        state_vector: np.ndarray,
        path_prob: float,
        emit_prob: float,
        path: Tuple[int, ...] = (),
    ):
        self.state_vector = state_vector  # The mixed state vector at this node
        self.path_prob = path_prob  # Probability of the path leading to this state
        self.emit_prob = emit_prob  # Probability of the emission leading to this state
        self.path = path  # The sequence of emissions leading to this state
        self.children: List[Mixed_State_Tree] = []  # Child mixed states

    def add_child(self, child: "Mixed_State_Tree"):
        self.children.append(child)

    def max_depth(self) -> int:
        if len(self.children) == 0:
            return 0
        else:
            return 1 + max([child.max_depth() for child in self.children])

    # Optional: Method to recursively print the tree for debugging or visualization
    def print_tree(self, level=0):
        indent = "  " * level
        print(
            f"{indent}- State: {self.path}, Path Prob: {self.path_prob:.4f}, Emit Prob: {self.emit_prob:.4f}"
        )
        for child in self.children:
            child.print_tree(level + 1)

    def get_node_by_path(self, search_path: Tuple[int]) -> Optional["Mixed_State_Tree"]:
        """
        Retrieve a node by its path. Returns None if the path does not exist.

        :param search_path: A tuple representing the path to search for.
        :return: Mixed_State_Tree node if found, else None.
        """
        # Base case: if the current node's path matches the search_path, return the current node
        if self.path == search_path:
            return self

        # If the search_path is longer than the current path, check if the beginning of search_path matches the current path
        if (
            len(search_path) > len(self.path)
            and search_path[: len(self.path)] == self.path
        ):
            # Iterate over children to find a matching path
            for child in self.children:
                # Recursively search in children
                result = child.get_node_by_path(search_path)
                if result is not None:
                    return result
        # Return None if the path was not found in the current subtree
        return None

    def get_belief_states(self) -> List[np.ndarray]:
        """
        Returns a list of belief states for all nodes in the tree.
        The belief states are the state vectors.
        """
        belief_states = [self.state_vector]  # Include the current node's state vector

        # Recursively collect belief states from children
        for child in self.children:
            belief_states.extend(child.get_belief_states())

        return belief_states
       