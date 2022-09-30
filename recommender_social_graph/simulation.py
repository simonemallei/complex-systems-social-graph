import networkx as nx
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import copy
from graph_creation import create_graph
from abeba_methods import simulate_epoch, compute_activation
from metrics import polarisation, sarle_bimodality, disagreement
from content.metrics import feed_entropy, feed_satisfaction
from content.content_recommender import simulate_epoch_content_recommender
import pandas as pd

def bootstrap(config):
    return create_graph(
        config["n_nodes"],
        config["beta"],
        avg_friend = config["avg_friend"],
        prob_post = config["prob_post"],
        hp_alpha = config["hp_alpha"],
        hp_beta = config["hp_beta"],
    )

def print_graph(G, title):
    print(title)
    colors = list(nx.get_node_attributes(G, 'opinion').values())
    labels =  nx.get_node_attributes(G, 'opinion')
    print(tabulate([[key] + [np.round(val, 3)] for key, val in labels.items()], headers=["node label", "opinion value"]))
    nx.draw(G, labels= dict([index for index in enumerate(labels)]), node_color=colors, font_color='darkturquoise', vmin=-1, vmax=1, cmap = plt.cm.get_cmap('magma'))
    plt.show()

def run_epochs(G, num_epochs, params, config):
    strategies = ["no_recommender", "random", "normal", "nudge", "similar", "unsimilar"]
    # Simulating an epoch and printing the opinion graph obtained
    data = {
        "strategy": [], 
        "epoch": [], 
        "satisfaction_mean": [], 
        "satisfaction_std": [], 
        "satisfaction_cov": []
    }
    graphs = {strategy : copy.deepcopy(G) for strategy in strategies}
    for i in range(num_epochs):
        for strategy in strategies:
            graphs[strategy] = simulate_epoch_content_recommender(
                G = graphs[strategy], 
                rate_updating_nodes = config["prob_act"], 
                epsilon = config["post_epsilon"],
                strategy = strategy,
                strat_param=params[strategy], 
                estim_strategy="kalman",
            )
        
        for text, curr_G in graphs.items():
            satisfaction_res = feed_satisfaction(curr_G)
            
            graphs[text] = curr_G
            data["strategy"].append(text)
            data["epoch"].append(i)
            
            if not(satisfaction_res == {}):
                data["satisfaction_mean"].append(np.mean(list(satisfaction_res.values())))
                data["satisfaction_std"].append(np.std(list(satisfaction_res.values())))
                sat_coverage = np.round(len(list(satisfaction_res.values())) / len(G.nodes()) * 100, 3)
                data["satisfaction_cov"].append(sat_coverage)
                #print(f"Satisfaction ({text} - mean): {np.mean(list(satisfaction_res.values()))}")
                #print(f"Satisfaction ({text} - std): {np.std(list(satisfaction_res.values()))}")
                #sat_coverage = np.round(len(list(satisfaction_res.values())) / len(G.nodes()) * 100, 3)
                #print(f"Satisfaction ({text} - coverage): {sat_coverage}%") 
            else:
                data["satisfaction_mean"].append(np.NaN)
                data["satisfaction_std"].append(np.NaN)
                data["satisfaction_cov"].append(0.0)
            
    df = pd.DataFrame(data=data)
    return graphs, df

def simulate(G, num_epochs, params, config):
    graphs, df = run_epochs(G, num_epochs, params, config)
    
    #print_graph(G, "Starting: ")
    
    #for strategy, graph in graphs.items():
    #    print_graph(graph, strategy)

    for strategy, graph in graphs.items():
        print(f"Satisfaction ({strategy} - mean): {np.mean(list(feed_satisfaction(graph).values()))}")
        print(f"Satisfaction ({strategy} - std): {np.std(list(feed_satisfaction(graph).values()))}")
    
    return df

def main():
    config = {
        "n_nodes" : 100, 
        "beta": [5], 
        "avg_friend": 10, 
        "prob_post": [0.5], 
        "hp_alpha": 2, 
        "hp_beta": 0.3,
        "prob_act": 0.5,
        "post_epsilon": 0.1,
    }

    num_epochs = 50

    random_param = {'n_post': 1}
    normal_param = {'normal_mean': 0.0, 'normal_std': 0.1, 'n_post': 1}
    nudge_param = {'nudge_goal': 0.0, 'n_post': 1}
    similar_param = {'similar_thresh': 0.5}
    unsimilar_param = {'unsimilar_thresh': 0.5}

    params = {
        "no_recommender": {},
        "random": random_param,
        "normal": normal_param,
        "nudge": nudge_param,
        "similar": similar_param,
        "unsimilar": unsimilar_param,
    }


    initial_graph = bootstrap(config)
    print_graph(initial_graph, "Starting graph: ")

    # From here you can adapt the code to print some graphs
    results_df = simulate(initial_graph, num_epochs, params, config)
    



if __name__ == "__main__":
    main()