import matplotlib.pyplot as plt
import networkx as nx
from entropy_analysis import compute_block_entropy, compute_conditional_entropy, compute_empirical_conditional_entropy
from typing import List

def visualize_graph(G: nx.DiGraph) -> None:
    """Visualize the provided graph."""
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels
    edge_labels = {(i, j): G[i][j]['label'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.show()

def visualize_graph_with_selective_offset(G: nx.DiGraph) -> None:
    """Visualize the graph with offset only for bidirectional edges."""
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    
    for edge in G.edges():
        if (edge[1], edge[0]) in G.edges():  # Check for reverse edge
            nx.draw_networkx_edges(G, pos, edgelist=[edge], connectionstyle="arc3,rad=0.1")
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[edge])
    
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels
    edge_labels = {(i, j): G[i][j]['label'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)
    
    plt.show()


def plot_block_entropy_diagram(sequence: List[int], max_block_length: int):
    """Plot the block entropy diagram."""
    block_entropies = compute_block_entropy(sequence, max_block_length)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_block_length + 1), block_entropies, marker='o', linestyle='-')
    plt.xlabel('Block Length (L)')
    plt.ylabel('Block Entropy H(L)')
    plt.title('Block Entropy Diagram')
    plt.grid(True)
    plt.show()

def plot_conditional_entropy_diagram(sequence: List[int], max_block_length: int):
    """Plot the conditional entropy diagram."""
    conditional_entropies = compute_conditional_entropy(sequence, max_block_length)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_block_length + 1), conditional_entropies, marker='o', linestyle='-')
    plt.xlabel('Block Length (L)')
    plt.ylabel('Conditional Entropy H(next symbol | previous L symbols)')
    plt.title('Conditional Entropy Diagram')
    plt.grid(True)
    plt.show()

def plot_empirical_conditional_entropy_diagram(sequence: List[int], max_block_length: int):
    """Plot the empirical conditional entropy diagram."""
    conditional_entropies = compute_empirical_conditional_entropy(sequence, max_block_length)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_block_length + 1), conditional_entropies, marker='o', linestyle='-')
    plt.xlabel('Block Length (L)')
    plt.ylabel('Empirical Conditional Entropy H(next symbol | previous L symbols)')
    plt.title('Empirical Conditional Entropy Diagram')
    plt.grid(True)
    plt.show()