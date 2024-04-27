import numpy as np
from numpy.typing import NDArray
import networkx as nx

from src.plotting import plot_graph, plot_loss_history


NDArrayInt = NDArray[np.int_]


def number_of_conflicts(G, colors):
    set_colors(G, colors)
    n = 0
    for n_in, n_out in G.edges:
        if G.nodes[n_in]["color"] == G.nodes[n_out]["color"]:
            n += 1
    return n


def set_colors(G, colors):
    for n, color in zip(G.nodes, colors):
        G.nodes[n]["color"] = color

def tweak_colors(colors, max_color):
    index_to_change = np.random.randint(len(colors))
    colors[index_to_change] = np.random.randint(max_color)
    return colors

def solve_via_simulated_annealing(G: nx.Graph, n_max_colors: int, initial_colors: NDArrayInt, n_iters: int):
    loss_history = np.zeros((n_iters,), dtype=np.int_)
    temperature = 1.0
    current_colors = initial_colors.copy()
    best_colors = initial_colors.copy()

    for i in range(n_iters):
        new_colors = tweak_colors(current_colors.copy(), n_max_colors)
        new_conflicts = number_of_conflicts(G, new_colors)
        current_conflicts = number_of_conflicts(G, current_colors)

        if new_conflicts < current_conflicts or np.random.rand() < np.exp((current_conflicts - new_conflicts) / temperature):
            current_colors = new_colors
            loss_history[i] = new_conflicts
        else:
            loss_history[i] = current_conflicts

        if current_conflicts < number_of_conflicts(G, best_colors):
            best_colors = current_colors

        temperature *= 0.99

    set_colors(G, best_colors)  

    return loss_history


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n=100, p=0.05, seed=seed)
    plot_graph(G)

    n_max_iters = 500
    n_max_colors = 3
    initial_colors = np.random.randint(low=0, high=n_max_colors - 1, size=len(G.nodes))

    loss_history = solve_via_simulated_annealing(
        G, n_max_colors, initial_colors, n_max_iters
    )
    plot_loss_history(loss_history)

