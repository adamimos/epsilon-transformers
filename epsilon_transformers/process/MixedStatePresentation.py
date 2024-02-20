from jaxtyping import Float
from typing import Set
import numpy as np
from uuid import UUID, uuid1

class MixedStateTreeNode:
	id: UUID
	state_vector: Float[np.ndarray, "n_states"] # Prob vector
	children: Set['MixedStateTreeNode']
	
	def __init__(self, state_vector: Float[np.ndarray, "n_states"], children: Set['MixedStateTreeNode']):
		self.id = uuid1()
		self.state_vector = state_vector
		self.children = children
	
class MixedStateTree:
	root_node: UUID
	process: str
	nodes: Set[MixedStateTreeNode]