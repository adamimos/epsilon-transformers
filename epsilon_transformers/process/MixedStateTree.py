from multiprocessing.util import sub_debug
from jaxtyping import Float
from typing import List, Set, Tuple
import numpy as np
from collections import deque
from scipy.stats import entropy


# TODO: Move the derive MSP function to be in the MSP init
# TODO: Add

class MixedStateTreeNode:
	state_prob_vector: Float[np.ndarray, "n_states"]
	path: List[int]
	children: Set['MixedStateTreeNode']
	emission_prob: float
	path_prob: float
	
	def __init__(self, state_prob_vector: Float[np.ndarray, "n_states"], children: Set['MixedStateTreeNode'], path: List[int], emission_prob: float, path_prob: float):
		self.state_prob_vector = state_prob_vector
		self.path = path
		self.children = children
		self.emission_prob = emission_prob
		self.path_prob = path_prob
	
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
	
	@property
	def paths(self) -> List[List[int]]:
		return [x.path for x in self.nodes]
	
	@property
	def paths_and_belief_states(self) -> List[Tuple[List[int], Float[np.ndarray, "n_states"]]]:
		paths_and_beliefs = [(x.path, x.state_prob_vector) for x in self.nodes]
		paths = [x[0] for x in paths_and_beliefs]
		beliefs = [x[1] for x in paths_and_beliefs]
		return paths, beliefs
	
	@property
	def block_entropy(self) -> Float[np.ndarray, "depth"]:
		depth_emission_probs = self._traverse(node=self.root_node, depth=0, accumulated_prob=1.0)
		block_entropy = np.array([entropy(probs) if probs else 0 for probs in depth_emission_probs])
		return block_entropy

	@property
	def myopic_entropy(self) -> Float[np.ndarray, "depth-1"]:
		return np.diff(self.block_entropy)
	
	def __init__(self, root_node: MixedStateTreeNode, process: str, nodes: Set[MixedStateTreeNode], depth: int):
		self.root_node = root_node
		self.process = process
		self.nodes = nodes
		self.depth = depth

	def _traverse(self, node: MixedStateTreeNode, depth: int, accumulated_prob: float) -> List[List[float]]:
		stack = deque([(node, depth, accumulated_prob)])
		depth_emission_probs = [[] for _ in range(self.depth)]
        
		while stack:
			node, depth, accumulated_prob = stack.pop()
			if depth < self.depth:
				if node is not self.root_node:
					depth_emission_probs[depth].append(accumulated_prob * node.emission_prob)
				for child in sorted(node.children, key=lambda x: tuple(x.path)):  # Ensure deterministic order by sorting
					stack.append((child, depth + 1, accumulated_prob * node.emission_prob if node is not self.root_node else 1.0))
		return depth_emission_probs

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
	

	def build_msp_transition_matrix(self) -> Float[np.ndarray, "num_emission num_msp_nodes num_msp_nodes"]:
		seen_prob_vectors = {}
		max_state_index = -1  # To keep track of the last index assigned to a unique state
		queue = deque([(self.root_node, None, -1, 0, 1.0)])  # (node, emitted_symbol, parent_state_index, emission_prob, path_prob)
		# get the number of symbols by looking at all entries of all paths and finding the max index
		num_symbols = len(np.unique(np.concatenate([np.unique(x) for x in self.paths])))
		num_nodes = len(self.nodes)
		
		# Assuming we know the number of symbols and maximum states to expect
		M = np.zeros((num_symbols, num_nodes, num_nodes))  # Adjust size appropriately

		while queue:
			current_node, emitted_symbol, from_state_index, emission_prob, path_prob = queue.popleft()
			rounded_vector = np.around(current_node.state_prob_vector, decimals=5)
			vector_tuple = tuple(rounded_vector.tolist())

			# Check if we've seen this state vector before
			if vector_tuple not in seen_prob_vectors:
				max_state_index += 1
				seen_prob_vectors[vector_tuple] = max_state_index
				to_state_index = max_state_index
			else:
				to_state_index = seen_prob_vectors[vector_tuple]

			# Only add this if there's a valid symbol and from_state_index
			if emitted_symbol is not None and from_state_index != -1:
				M[emitted_symbol, from_state_index, to_state_index] = emission_prob

			# Check and add children to the queue
			for child in current_node.children:
				if child.path:
					child_symbol = child.path[-1]  # Assume last element of the path is the symbol
					child_emission_prob = child.emission_prob
					child_path_prob = path_prob * child_emission_prob
					queue.append((child, child_symbol, to_state_index, child_emission_prob, child_path_prob))

		# delete entries that were never visited
		M = M[:, :max_state_index + 1, :max_state_index + 1]

		return M
	
	def _get_nodes_at_depth(self, depth: int) -> Set[MixedStateTreeNode]:
		return {n for n in self.nodes if len(n.path) == depth}
	
	def get_paths_and_probs(self, depth: int) -> List[Tuple[List[int], float]]:
		nodes = self._get_nodes_at_depth(depth)
		paths = [x.path for x in nodes]
		prbs = [x.path_prob for x in nodes]
		return paths, prbs