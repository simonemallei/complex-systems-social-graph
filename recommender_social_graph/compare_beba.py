import copy
import json
import os
from collections import OrderedDict
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import abeba_methods
import beba_methods
from graph_creation import create_graph
from metrics import polarisation, sarle_bimodality, disagreement, echo_chamber_value
from util import track_time

_STUBBORNESS = 0
_EPSILON = 0.0

"""
_simulate_beba simulates an epoch with BEBA frameworks.

Parameters
----------
    G : {networkx.Graph}
        The graph
    nodes : {list}
        Contains the nodes to update
Returns
-------
  G : {networkx.Graph}
      The updated graph.
"""


def _simulate_beba(G, nodes):
    G = beba_methods.compute_update(G=G, n_update_list=nodes)
    return G


"""
_simulate_abeba simulates an epoch with ABEBA frameworks.

Parameters
----------
    G : {networkx.Graph}
        The graph
    nodes : {list}
        Contains the nodes to update
Returns
-------
  G : {networkx.Graph}
      The updated graph.
"""


def _simulate_abeba(G, nodes):
    G = abeba_methods.compute_activation(G=G, nodes=nodes, stubborness=_STUBBORNESS)
    G, _ = abeba_methods.compute_post(G=G, nodes=nodes, epsilon=_EPSILON)

    return G


"""
compute_metrics calculates the metrics in the current epoch.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.

Returns
-------
    epoch_metrics : {dictionary}
        It contains values of all metrics calculated in the current epoch.
"""


def _compute_metrics(G):
    epoch_metrics = {}
    epoch_metrics["polarization_value"] = polarisation(G)
    epoch_metrics["bimodality"] = sarle_bimodality(G)

    echo_chamber_value_result = {}
    (
        echo_chamber_value_result["single_values"],
        echo_chamber_value_result["mean"],
    ) = echo_chamber_value(G)
    epoch_metrics["echo_chamber_value"] = echo_chamber_value_result

    disagreement_result = {}
    disagreement_result["mean"], disagreement_result["variance"] = disagreement(G)
    epoch_metrics["disagreement"] = disagreement_result

    epoch_metrics["engagement"] = nx.get_node_attributes(G, name="engagement")

    return epoch_metrics


"""
simulate_epoch simulates a epoch with ABEBA and BEBA frameworks.

Parameters
----------
    G_abeba : {networkx.Graph}
        The graph simulated with ABEBA
    G_beba : {networkx.Graph}
        The graph simulated with BEBA
    rate_updating_nodes : {float}
        The percentage of the nodes that will be activated. Interval [0,1]
Returns
-------
  G : {networkx.Graph}
      The updated graph.
"""


def _simulate_epoch(G_abeba, G_beba, rate_updating_nodes):
    # Sampling randomly the activating nodes
    updating_nodes = int(rate_updating_nodes * len(G_abeba.nodes()))
    act_nodes = np.random.choice(
        range(len(G_abeba.nodes())), size=updating_nodes, replace=False
    )

    G_abeba = _simulate_abeba(G=G_abeba, nodes=act_nodes)
    G_beba = _simulate_beba(G=G_beba, nodes=act_nodes)

    return G_abeba, G_beba


"""
simulate_epochs first saves the initial opinions. Then a cycle is made on each epoch, 
in which the simulate_epoch_content_people_recommender method is called first, which 
simulates a single epoch, and then the compute_metrics method, which calculates the 
metrics on the current epoch.

Parameters
----------
    G_abeba : {networkx.Graph}
        The graph simulated with ABEBA
    G_beba : {networkx.Graph}
        The graph simulated with BEBA
    rate_updating_nodes : {float}
        The percentage of the nodes that will be activated. Interval [0,1]
    num_epochs : {int}
        The number of epochs to simulate
Returns
-------
    G, initial_opinions, opinions_and_metrics : {tuple}
        A tuple containing the final graph, the initial views, and the views and metrics for each epoch
"""


def _simulate_epochs(G_abeba, G_beba, rate_updating_nodes, num_epochs):
    results_abeba = {
        "graph": G_abeba,
        "initial_opinions": nx.get_node_attributes(G_abeba, "opinion"),
        "opinions_and_metrics": [],
    }
    results_beba = {
        "graph": G_beba,
        "initial_opinions": nx.get_node_attributes(G_beba, "opinion"),
        "opinions_and_metrics": [],
    }

    for i in range(num_epochs):
        with track_time(msg=f"simulating epoch number {i}"):
            results_abeba["graph"], results_beba["graph"] = _simulate_epoch(
                G_abeba=G_abeba,
                G_beba=G_beba,
                rate_updating_nodes=rate_updating_nodes,
            )

            epoch_metrics = _compute_metrics(G_abeba)
            epoch_data = {
                "metrics": epoch_metrics,
                "opinions": nx.get_node_attributes(G_abeba, "opinion"),
            }
            results_abeba["opinions_and_metrics"].append(epoch_data)

            epoch_metrics = _compute_metrics(G_beba)
            epoch_data = {
                "metrics": epoch_metrics,
                "opinions": nx.get_node_attributes(G_beba, "opinion"),
            }
            results_beba["opinions_and_metrics"].append(epoch_data)

    return results_abeba, results_beba


"""
save_img_graph saves a graph in the file system

Parameters
----------
    G : {networkx graph}
        A networkx graph
    title : {String}
        The name of the file where the graph will be saved
    path : {String}
        The path where the graph will be saved

Returns
-------

"""


def save_img_graph(G, title, path):
    colors = list(nx.get_node_attributes(G, "opinion").values())
    labels = nx.get_node_attributes(G, "opinion")
    nx.draw(
        G,
        labels=dict([index for index in enumerate(labels)]),
        node_color=colors,
        font_color="darkturquoise",
        vmin=-1,
        vmax=1,
        cmap=plt.cm.get_cmap("magma"),
    )
    plt.savefig(path + "/" + title + ".png", format="PNG")
    plt.clf()


"""
_save_results creates a counter for each configuration group. It will be used to make unique 
the name of the files in which the results of each expanded configuration will be saved. Then, for each 
element of the result list, the path where to save the data is retrieved from the graph_group_id. Then 
an ordered dictionary is created in which the first element will be the object of the initial opinions 
of each node (so that it can always appear at the beginning of the json file), and the next one containing 
all the results on the various epochs. Finally, the json file with the data structure is created and saved 
and the counter relative to the current conf_group_id is modified.

Parameters
----------
    folders_dict : {dict}
        A dictionary in which the keys are the ids of the configuration groups and values are all output 
        folders paths that have been created to hold simulation results

Returns
-------
"""


def _save_results(output_path, results):
    now_timestamp = datetime.utcnow()
    abs_path = f"{output_path}/{now_timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_config"
    os.mkdir(abs_path)
    for idx, result in enumerate(results):
        abeba, beba = result[0], result[1]

        save_img_graph(
            abeba["graph"],
            f"Final_graph_{idx}_ABEBA",
            abs_path,
        )
        save_img_graph(
            beba["graph"],
            f"Final_graph_{idx}_BEBA",
            abs_path,
        )

        result_dict = OrderedDict()

        result_dict["abeba"] = {
            "initial_opinions": abeba["initial_opinions"],
            "epochs_data": abeba["opinions_and_metrics"],
        }
        result_dict["beba"] = {
            "initial_opinions": beba["initial_opinions"],
            "epochs_data": beba["opinions_and_metrics"],
        }

        # Saving model configuration (for the entire group)
        with open(
            f"{abs_path}/model_configuration_result{idx}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)


def _multi_run_simulate(args):
    return _simulate_epochs(*args)


def compare_beba_abeba():
    OUTPUT_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/beba_comparison"
    NODES = 100
    AVG_FRIENDS = 50
    BETA = [1]
    PROB_POST = [0.5]
    HP_ALPHA = 2
    HP_BETA = 1
    RATE = 0.5
    EPOCHS = 100
    RUNS = 8
    G = create_graph(
        n_ag=NODES,
        avg_friend=AVG_FRIENDS,
        beba_beta=BETA,
        prob_post=PROB_POST,
        hp_alpha=HP_ALPHA,
        hp_beta=HP_BETA,
    )

    iterable_for_multiproc = list(
        (copy.deepcopy(G), copy.deepcopy(G), RATE, EPOCHS) for _ in range(RUNS)
    )

    pool = Pool()
    results = tuple(pool.imap_unordered(_multi_run_simulate, iterable_for_multiproc))
    _save_results(output_path=OUTPUT_PATH, results=results)


def main():
    compare_beba_abeba()


if __name__ == "__main__":
    main()
