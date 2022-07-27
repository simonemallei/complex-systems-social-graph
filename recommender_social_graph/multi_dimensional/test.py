from cProfile import label
import matplotlib
from tabulate import tabulate
from graph_creation import *
from abeba_methods import *
from metrics import polarisation
import matplotlib.pyplot as plt

"""
    print_graph prints an opinion graph where there are 3 opinions 
    and uses the value of opinions to choose the RGB color of a node
"""
def print_graph(G, print_labels=True):
    labels = nx.get_node_attributes(G, 'opinion')
    label_map = {}
    color_map = []
    for label in labels:
        label_map[label] = [round(x, 2) for x in labels[label].tolist()]   
        red = (label_map[label][0] + 1) / 2
        green = (label_map[label][1] + 1) / 2
        blue = (label_map[label][2] + 1) / 2
        color_map.append(matplotlib.colors.to_hex([red, green, blue]))
    if print_labels is True:
        nx.draw(G, labels=label_map, node_color=color_map, with_labels=True)
    else:
        nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

nodes, ops = 50, 3
G = create_graph(nodes, ops, [1], avg_friend = 2, hp_alpha=5, hp_beta=0)
G = apply_initial_feed(G, ops, n_post=2)
print_graph(G, print_labels=False)
# print(tabulate([[key] + [np.round(val, 3)] for key, val in labels.items()], headers=["node label", "opinion value"]))
for i in range(500):
    G = simulate_epoch_updated(G, ops, 50, 50)
print_graph(G, print_labels=False)