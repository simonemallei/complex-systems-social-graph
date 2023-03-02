import copy
import networkx as nx

from typing import Dict


class RunSimulator:
    """Component that permits to simulate a run."""

    def __init__(self, initial_graph, run):
        self._run = run
        self._G = self._setup_graph(data=initial_graph)
    
    def simulate_epochs(self) -> Dict[int, nx.Graph]:
        """
        Simulate the run.

        :return: a generator that yield for each epoch its graph
        """
        graphs = {}
        G = copy.deepcopy(self._G)
        for idx, epoch in enumerate(self._run["epochs_data"]):

            edges_to_add = tuple((int(edge[0]), int(edge[1])) for edge in epoch["graph_changes"]["edges_added"].items())
            G.add_edges_from(edges_to_add)

            edges_to_delete = tuple((int(edge[0]), int(edge[1])) for edge in epoch["graph_changes"]["edges_deleted"].items())
            G.remove_edges_from(edges_to_delete)

            new_opinions = {node: epoch["opinions"][str(node)] for node in G.nodes()}
            nx.set_node_attributes(G, new_opinions, 'opinion')

            yield G

    def _setup_graph(self, data) -> nx.Graph:
        """
        Setup the initial graph from the given data.
        
        :param data: initial graph data
        :return: the corresponding graph
        """

        G = nx.Graph()

        nodes = range(data["configuration"]["n_nodes"])
        G.add_nodes_from(nodes)
        
        edges = tuple((edge["node_1"], edge["node_2"]) for edge in data["edges"])
        G.add_edges_from(edges)

        opinions = {node: self._run["initial_opinions"][str(node)] for node in G.nodes()}
        nx.set_node_attributes(G, opinions, 'opinion')

        return G
