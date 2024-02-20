from jaxtyping import Float
from typing import Set
import numpy as np

class MixedStateTreeNode:
	state_prob_vector: Float[np.ndarray, "n_states"]
	children: Set['MixedStateTreeNode']
	
	def __init__(self, state_prob_vector: Float[np.ndarray, "n_states"], children: Set['MixedStateTreeNode']):
		self.state_prob_vector = state_prob_vector
		self.children = children
		
	def add_child(self, child: 'MixedStateTreeNode'):
		self.children.add(child)

class MixedStateTree:
	root_node: MixedStateTreeNode
	process: str
	nodes: Set[MixedStateTreeNode]
	
	def __init__(self, root_node: MixedStateTreeNode, process: str, nodes: Set[MixedStateTreeNode]):
		self.root_node = root_node
		self.process = process
		self.nodes = nodes