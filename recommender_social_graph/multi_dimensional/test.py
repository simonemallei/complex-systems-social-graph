from tabulate import tabulate
from graph_creation import *
from abeba_methods import *
from metrics import polarisation


nodes, ops = 5, 3
G = create_graph(nodes, ops, [1], avg_friend = 3, hp_alpha=5, hp_beta=0)
G = apply_initial_feed(G, ops, n_post=2)
# print("Starting graph: ")
labels =  nx.get_node_attributes(G, 'opinion')
# print(tabulate([[key] + [np.round(val, 3)] for key, val in labels.items()], headers=["node label", "opinion value"]))
for i in range(50):
    G = simulate_epoch_updated(G, ops, 50, 50)
labels =  nx.get_node_attributes(G, 'opinion')
# print(tabulate([[key] + [np.round(val, 3)] for key, val in labels.items()], headers=["node label", "opinion value"]))
print(polarisation(G))