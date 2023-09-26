import numpy as np
from typing import List, Tuple
from collections import Counter

def compute_block_entropy(sequence: List[int], max_block_length: int) -> np.ndarray:
    """Compute the block entropy for blocks of varying lengths."""
    block_entropies = []
    
    for L in range(1, max_block_length + 1):
        # Create blocks of length L
        blocks = [tuple(sequence[i:i+L]) for i in range(len(sequence) - L + 1)]
        
        # Compute the probability distribution of the blocks
        block_counts = Counter(blocks)
        total_blocks = len(blocks)
        block_probs = {block: count / total_blocks for block, count in block_counts.items()}
        
        # Compute the block entropy H(L)
        entropy_L = -sum(p * np.log2(p) for p in block_probs.values())
        block_entropies.append(entropy_L)
    
    return np.array(block_entropies)

def compute_conditional_entropy(sequence: List[int], max_block_length: int) -> np.ndarray:
    """Compute the conditional entropy H(next symbol | previous L symbols) for varying L."""
    conditional_entropies = []
    
    # First, compute the block entropies for all required lengths (up to L+1)
    all_block_entropies = compute_block_entropy(sequence, max_block_length + 1)
    
    for L in range(1, max_block_length + 1):
        joint_entropy = all_block_entropies[L]  # Joint entropy for L+1 symbols
        block_entropy = all_block_entropies[L-1]  # Block entropy for L symbols
        
        # Compute conditional entropy
        conditional_entropy = joint_entropy - block_entropy
        conditional_entropies.append(conditional_entropy)
    
    return np.array(conditional_entropies)

def compute_empirical_conditional_entropy(sequence: List[int], max_block_length: int) -> List[float]:
    """Compute the empirical conditional entropy H(next symbol | previous L symbols) for varying L."""
    NUM_SYMBOLS = 2
    conditional_entropies = []

    for L in range(1, max_block_length + 1):
        # Dictionary to store counts of observed blocks of length L followed by a symbol
        block_followed_by_symbol_counts = Counter([(tuple(sequence[i:i+L]), sequence[i+L]) for i in range(len(sequence) - L)])
        
        # Dictionary to store counts of observed blocks of length L
        block_counts = Counter([tuple(sequence[i:i+L]) for i in range(len(sequence) - L + 1)])
        
        # Conditional entropy computation
        entropy = 0
        for block, block_count in block_counts.items():
            for symbol in range(NUM_SYMBOLS):
                # Empirical conditional probability p(symbol | block)
                conditional_prob = block_followed_by_symbol_counts.get((block, symbol), 0) / block_count
                if conditional_prob > 0:
                    entropy -= (block_count / len(sequence)) * conditional_prob * np.log(conditional_prob)
        
        conditional_entropies.append(entropy)

    return conditional_entropies