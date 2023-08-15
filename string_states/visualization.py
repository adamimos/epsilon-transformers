import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from finite_state_machine import FiniteStateMachine

def plot_from_transition_matrix(fsm: FiniteStateMachine, layout: str = 'circular'):
    matrix = fsm.transition_matrix
    G = nx.DiGraph()

    # Add nodes
    for state in fsm.states:
        G.add_node(state)

    # Add edges
    for key, value in fsm.transition_function.items():
        from_state = key[:-1]
        to_state = value
        output = int(key[-1])
        weight = matrix[fsm.states.index(to_state), fsm.states.index(from_state)]

        if weight != 0:  # If there's a transition from from_state to to_state
            G.add_edge(from_state, to_state, weight=weight, label=output)

    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        AssertionError("Invalid layout")


    # Get edge weights and normalize for drawing
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # edge color is blue if output is 0, red if output is 1
    edge_colors = ['b' if G[u][v]['label'] == 0 else 'r' for u, v in G.edges()]

    # Plot nodes
    nx.draw_networkx_nodes(G, pos, node_color='k', node_size=500)

    # if labels are long, make the text smaller to fit in the node
    state_string_len = len(fsm.states[0])
    if state_string_len > 3:
        font_size = 8 - (state_string_len - 3)
    else:
        font_size = 8
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=font_size)

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
