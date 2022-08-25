from cProfile import label
from cgi import test
import matplotlib
from tabulate import tabulate
from graph_creation import *
from abeba_methods import *
from metrics import *
import matplotlib.pyplot as plt
from content.content_recommender import *
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


def test_graph():
    nodes, ops = 50, 3
    G = create_graph(nodes, ops, [1], avg_friend = 2, hp_alpha=5, hp_beta=0)
    print_graph(G, False)
    random_param = {'n_post': 2}
    for i in range(500):
        G = simulate_epoch_content_recommender(G, ops, 50, 50, strat_param=random_param)
        # simulate_epoch_updated(G, ops, 50, 50, 0)
    print_graph(G, False)


def main():
    test_graph()
    return 
    nodes, ops = 50, 3
    G = create_graph(nodes, ops, [1], avg_friend = 2, hp_alpha=5, hp_beta=0)
    G = apply_initial_feed(G, ops, n_post=2)
    print_graph(G, False)

    random_param = {'n_post': 2}
    normal_param = {'normal_mean': 0.5, 'normal_std': 0.1, 'n_post': 2}
    nudge_param = {'nudge_goal': np.array([0.5, 0.5, 0.5]), 'n_post': 2}
    similar_param = {'similar_thresh': 0.5}
    unsimilar_param = {'unsimilar_thresh': 0.2}

    steps = 200
    for i in range(steps):
        #G = simulate_epoch_content_recommender(G, ops, 50, 50, strat_param=random_param)
        #G = simulate_epoch_content_recommender(G, ops, 50, 50, strategy="normal", strat_param=normal_param)
        #G = simulate_epoch_content_recommender(G, ops, 50, 50, strategy="nudge",strat_param=nudge_param)
        #G = simulate_epoch_content_recommender(G, ops, 50, 50,strategy="nudge_opt",strat_param=nudge_param)
        G = simulate_epoch_content_recommender(G, ops, 50, 50, strategy="similar", strat_param=similar_param)
        #G = simulate_epoch_content_recommender(_G, ops, 50, 50, strategy="unsimilar",strat_param=unsimilar_param)

    print_graph(G, False)
    print("Polarisation: " + str(disagreement(G)))
    #print(disagreement(G))
    #print(sarle_bimodality(G, ops))
    #print(feed_entropy(G, ops))

if __name__ == "__main__":
    main()

