from jaxtyping import Float
from typing import Set
import numpy as np
from uuid import UUID

class MixedStateTreeNode:
	id: UUID
	state_vector: Float[np.ndarray, "n_states"] # Prob vector
	children: Set['MixedStateTreeNode']
	
class MixedStateTree:
	root_node: UUID
	process: str
	nodes: Set[MixedStateTreeNode]