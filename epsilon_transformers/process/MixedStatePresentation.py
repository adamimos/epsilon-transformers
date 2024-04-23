from multiprocessing.util import sub_debug
from jaxtyping import Float
from typing import List, Set
import numpy as np

# TODO: Move the derive MSP function to be in the MSP init
# TODO: Add

class MixedStateTreeNode:
	state_prob_vector: Float[np.ndarray, "n_states"]
	path: List[int]
	children: Set['MixedStateTreeNode']
	
	def __init__(self, state_prob_vector: Float[np.ndarray, "n_states"], children: Set['MixedStateTreeNode'], path: List[int]):
		self.state_prob_vector = state_prob_vector
		self.path = path
		self.children = children
		
	def add_child(self, child: 'MixedStateTreeNode'):
		self.children.add(child)

class MixedStateTree:
	root_node: MixedStateTreeNode
	process: str
	depth: int
	nodes: Set[MixedStateTreeNode]

	@property
	def belief_states(self) -> Set[Float[np.ndarray, "num_states"]]:
		return [x.state_prob_vector for x in self.nodes]
	
	def __init__(self, root_node: MixedStateTreeNode, process: str, nodes: Set[MixedStateTreeNode], depth: int):
		self.root_node = root_node
		self.process = process
		self.nodes = nodes
		self.depth = depth

	def path_to_beliefs(self, path: List[int]) -> Float[np.ndarray, "path_length n_states"]:
		assert len(path) <= self.depth, f"path length: {len(path)} is too long . Tree has depth of {self.depth}"

		belief_states = []
		current_node = self.root_node
		for i in range(len(path)):
			sub_path = path[:i + 1]
			for child in current_node.children:
				if child.path == sub_path:
					belief_states.append(child.state_prob_vector)
					current_node = child
					break

		assert current_node.path == path, f"{path} is not a valid path for this process"
		assert len(belief_states) == len(path)
		return np.stack(belief_states)