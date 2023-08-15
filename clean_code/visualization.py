import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from finite_state_machine import FiniteStateMachine


def plot_from_transition_matrix(fsm: FiniteStateMachine):
    matrix = fsm.transition_matrix
    G = nx.DiGraph()

    # Add nodes
    for i in range(len(matrix)):
        G.add_node(i)

    # Add edges
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != 0:  # If there's a transition from i to j
                output = fsm.transition_output_matrix[i][j]
                G.add_edge(j,i, weight=matrix[i][j], label=output)

    pos = nx.circular_layout(G)

    # Get edge weights and normalize for drawing
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # edge color is blue if output is 0, red if output is 1
    edge_colors = ['b' if G[u][v]['label'] == 0 else 'r' for u, v in G.edges()]

    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=500)
    nx.draw_networkx_labels(G, pos, font_color='white')

    # Plot edges with alpha values proportional to the weights
    for i in range(len(weights)):
        edge = list(G.edges())[i]
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=edge_colors[i],
                               width=2, alpha=weights[i], connectionstyle='arc3, rad=0.1')

    # add a key for the edge label colors
    red_patch = mpatches.Patch(color='red', label='Output = 1')
    blue_patch = mpatches.Patch(color='blue', label='Output = 0')
    plt.legend(handles=[red_patch, blue_patch])

    plt.show()