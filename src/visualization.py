import matplotlib.pyplot as plt
import networkx as nx
from entropy_analysis import compute_block_entropy, compute_conditional_entropy, compute_empirical_conditional_entropy
from typing import List
from matplotlib import colors


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

def visualize_graph_with_selective_offset(G: nx.DiGraph, layout: str = 'spring', draw_edge_labels: bool = True, draw_color: bool = False, draw_mixed_state: bool = False) -> None:
    """Visualize the graph with offset only for bidirectional edges."""
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        raise ValueError(f"Invalid layout type: {layout}")

    # Compute the strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    recurrent_edges = [(u, v) for scc in sccs for u in scc for v in G.successors(u) if v in scc]
    transitory_edges = [edge for edge in G.edges() if edge not in recurrent_edges]
    transitory_states = [node for node in G.nodes() if node not in [edge[0] for edge in recurrent_edges]]

    # Draw nodes with double outline for the first state if draw_mixed_state is True
    for node in G.nodes():
        if draw_mixed_state:
            node_color = 'green' if node in transitory_states else 'purple'
            if node == 0:  # Check if the node is the first state
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color, edgecolors='black', linewidths=5)
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color)
        else:
            node_color = 'black'
            nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color)

    for edge in G.edges(data=True):
        if draw_color:
            edge_color = 'blue' if edge[2]['label'] == '0' else 'red'
            edge_alpha = edge[2]['weight']
        else:
            edge_color = 'black'
            edge_alpha = 1
        if (edge[1], edge[0]) in G.edges():  # Check for reverse edge
            nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], connectionstyle="arc3,rad=0.1", edge_color=edge_color, alpha=edge_alpha)
        else:
            if draw_mixed_state and ((edge[0], edge[1]) in transitory_edges):
                nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], edge_color=edge_color, alpha=edge_alpha, style='dotted')
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(edge[0], edge[1])], edge_color=edge_color, alpha=edge_alpha)
    
    nx.draw_networkx_labels(G, pos)

    # Draw edge labels
    if draw_edge_labels:
        edge_labels = {(i, j): f"{G[i][j]['label']}|{round(G[i][j]['weight']*100)}%" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)

    # Draw color legend if draw_color is True
    if draw_color:
        plt.plot([], [], color='blue', label='0')
        plt.plot([], [], color='red', label='1')
        plt.legend(title='Emission', loc='upper right')

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