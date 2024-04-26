import matplotlib.pyplot as plt
import networkx as nx
from epsilon_transformers.analysis.entropy_analysis import (
    compute_block_entropy,
    compute_conditional_entropy,
    compute_empirical_conditional_entropy,
)
from typing import List
from matplotlib import colors
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict
import numpy as np
from jaxtyping import Float

def determine_layout(G, layout_type):
    """Determine the layout of the graph."""
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "random": nx.random_layout,
        "shell": nx.shell_layout,
        "spectral": nx.spectral_layout,
        "hierarchical": hierarchical_layout,
        "dot": lambda G: nx.nx_agraph.graphviz_layout(G, prog="dot"),
        "neato": lambda G: nx.nx_agraph.graphviz_layout(G, prog="neato"),
        "fdp": lambda G: nx.nx_agraph.graphviz_layout(G, prog="fdp"),
        "sfdp": lambda G: nx.nx_agraph.graphviz_layout(G, prog="sfdp"),
        "twopi": lambda G: nx.nx_agraph.graphviz_layout(G, prog="twopi"),
        "circo": lambda G: nx.nx_agraph.graphviz_layout(G, prog="circo"),
    }

    layout_func = layout_funcs.get(layout_type)
    if layout_func is None:
        raise ValueError(f"Invalid layout type: {layout_type}")

    return layout_func(G)


def hierarchical_layout(G):
    """Generate a hierarchical layout for the graph."""
    # This is a simple hierarchical layout and might need adjustments based on the exact graph structure
    levels = {}
    for node in G.nodes():
        if node not in levels:
            levels[node] = 0
        for child in G.successors(node):
            levels[child] = levels[node] + 1

    max_level = max(levels.values())
    pos = {}
    for node, level in levels.items():
        # Distribute nodes evenly across the level
        level_nodes = [n for n, l in levels.items() if l == level]
        level_width = len(level_nodes)
        idx = level_nodes.index(node)
        x = idx - level_width / 2
        y = max_level - level
        pos[node] = (x, y)

    return pos


def get_colors():
    """Return the colors used in the graph."""
    return {
        "transitory": "#A1D4C0",
        "recurrent": "#B2C9F2",
        "observation_induced": "#D4B2E2",
        "edge_standard": "black",
    }


def draw_nodes(G, pos, transitory_states, colors, draw_mixed_state):
    """Draw nodes on the graph."""
    for node in G.nodes():
        if draw_mixed_state:
            node_color = (
                colors["transitory"]
                if node in transitory_states
                else colors["recurrent"]
            )
            if node == 0:  # Check if the node is the first state
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[node],
                    node_color=node_color,
                    edgecolors="black",
                    linewidths=2,
                )  # Primary circle with thicker outline
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[node],
                    node_color=node_color,
                    edgecolors="black",
                    linewidths=0.5,
                    node_size=325,
                )  # Secondary circle
            else:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[node], node_color=node_color, edgecolors="black"
                )
        else:
            node_color = colors["edge_standard"]
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node], node_color=node_color, edgecolors="black"
            )


def draw_edges(G, pos, transitory_edges, colors, draw_color, draw_mixed_state):
    """Draw edges on the graph."""
    for idx, edge in enumerate(G.edges(data=True)):
        edge_color = colors["edge_standard"]
        if draw_color:
            edge_color = "blue" if edge[2]["label"] == "0" else "red"
        edge_alpha = edge[2]["weight"]

        if (edge[1], edge[0]) in G.edges():
            # If there are multiple edges, adjust the radian value to separate the arcs
            num_edges = G.number_of_edges(edge[0], edge[1])
            rad = 0.3
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f"arc3,rad={rad}",
                edge_color=edge_color,
                alpha=edge_alpha,
            )
        elif G.number_of_edges(edge[0], edge[1]) > 1:
            # If there are multiple edges, adjust the radian value to separate the arcs
            num_edges = G.number_of_edges(edge[0], edge[1])
            rad = 0.3 * ((idx % num_edges) - (num_edges - 1) / num_edges)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(edge[0], edge[1])],
                connectionstyle=f"arc3,rad={rad}",
                edge_color=edge_color,
                alpha=edge_alpha,
            )
        else:
            if draw_mixed_state and ((edge[0], edge[1]) in transitory_edges):
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(edge[0], edge[1])],
                    edge_color=edge_color,
                    alpha=edge_alpha,
                    style="dotted",
                )
            else:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=[(edge[0], edge[1])],
                    edge_color=edge_color,
                    alpha=edge_alpha,
                )


def identify_recurrent_states(G):
    """Identify the recurrent states of the graph."""
    sccs = list(nx.strongly_connected_components(G))

    recurrent_states = set()
    for scc in sccs:
        is_recurrent = True
        for node in scc:
            for successor in G.successors(node):
                if successor not in scc:
                    is_recurrent = False
                    break
            if not is_recurrent:
                break
        if is_recurrent:
            recurrent_states.update(scc)

    return recurrent_states


def visualize_graph(
    G,
    layout="spring",
    draw_edge_labels=True,
    draw_color=False,
    draw_mixed_state=False,
    pdf=None,
):
    pos = determine_layout(G, layout)

    recurrent_states = identify_recurrent_states(G)
    transitory_states = [node for node in G.nodes() if node not in recurrent_states]
    # Identify recurrent edges
    recurrent_edges = [
        (u, v) for u in recurrent_states for v in recurrent_states if G.has_edge(u, v)
    ]
    transitory_edges = [edge for edge in G.edges() if edge not in recurrent_edges]

    colors = get_colors()

    draw_nodes(G, pos, transitory_states, colors, draw_mixed_state)
    draw_edges(G, pos, transitory_edges, colors, draw_color, draw_mixed_state)
    nx.draw_networkx_labels(G, pos, font_color="black")  # Adjusted font color to red

    if draw_edge_labels:
        edge_labels = {}
        for i, j, data in G.edges(data=True):
            edge_labels[(i, j)] = (
                f"$\mathbf{{{data['label']}}}$:{round(data['weight']*100)}%"
            )

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8
        )

    if pdf is not None:
        with plt.rc_context({"pdf.fonttype": 42}):
            plt.savefig(pdf, format="pdf")
            plt.show()
    else:
        plt.show()


def plot_block_entropy_diagram(sequence: List[int], max_block_length: int):
    """Plot the block entropy diagram."""
    block_entropies = compute_block_entropy(sequence, max_block_length)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_block_length + 1), block_entropies, marker="o", linestyle="-")
    plt.xlabel("Block Length (L)")
    plt.ylabel("Block Entropy H(L)")
    plt.title("Block Entropy Diagram")
    plt.grid(True)
    plt.show()


def plot_conditional_entropy_diagram(sequence: List[int], max_block_length: int):
    """Plot the conditional entropy diagram."""
    conditional_entropies = compute_conditional_entropy(sequence, max_block_length)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, max_block_length + 1), conditional_entropies, marker="o", linestyle="-"
    )
    plt.xlabel("Block Length (L)")
    plt.ylabel("Conditional Entropy H(next symbol | previous L symbols)")
    plt.title("Conditional Entropy Diagram")
    plt.grid(True)
    plt.show()


def plot_empirical_conditional_entropy_diagram(
    sequence: List[int], max_block_length: int
):
    """Plot the empirical conditional entropy diagram."""
    conditional_entropies = compute_empirical_conditional_entropy(
        sequence, max_block_length
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, max_block_length + 1), conditional_entropies, marker="o", linestyle="-"
    )
    plt.xlabel("Block Length (L)")
    plt.ylabel("Empirical Conditional Entropy H(next symbol | previous L symbols)")
    plt.title("Empirical Conditional Entropy Diagram")
    plt.grid(True)
    plt.show()

def transition_matrix_to_graph(transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"],
                               state_names: Optional[Dict[str, int]] = None) -> nx.DiGraph:
    """
    Convert a transition matrix to a graph.

    Parameters:
    transition_matrix (np.ndarray): The transition matrix of shape (n_outputs, n_states, n_states).
                                    n_states is the number of states in the machine.
                                    transition_matrix[i, j, k] is the probability of transitioning from state j to state k on output i.
    state_names (Dict[str, int], optional): A dictionary mapping state names to state indices.

    Returns:
    nx.DiGraph: The graph representation of the transition matrix.
    """
    # Get the number of outputs and states
    n_outputs, n_states, _ = transition_matrix.shape

    # Create an empty directed graph
    G = nx.MultiDiGraph()

    # Invert the state_names dictionary if it's provided
    if state_names:
        state_names = {v: k for k, v in state_names.items()}

    # Add nodes to the graph
    for i in range(n_states):
        node_label = state_names[i] if state_names else i
        G.add_node(node_label)

    # Add edges to the graph for each transition in the epsilon machine
    for i in range(n_outputs):
        for j in range(n_states):
            for k in range(n_states):
                # Add an edge from state j to state k with label i and weight equal to the transition probability
                # only if the transition probability is not zero
                if transition_matrix[i, j, k] != 0:
                    from_node = state_names[j] if state_names else j
                    to_node = state_names[k] if state_names else k
                    G.add_edge(from_node, to_node, label=str(i), weight=transition_matrix[i, j, k])

    return G
