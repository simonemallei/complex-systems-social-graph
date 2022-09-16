from venv import create
import networkx as nx
import random
from collections import defaultdict
import numpy as np
import math

"""
    Return homophilic random graph using BA preferential attachment model.
    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which 
    depends on the connectivity of sites and the homophily(similarities).
    homophily varies ranges from 0 to 1.

Parameters
----------
    N : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    ops : int
        Number of opinions of each node
"""

def MY_homophilic_ba_graph(N, m, ops=1, alpha=2, beta=1):
    G = nx.Graph()
    node_attribute = {}
    
    for n in range(N):
        # Generate opinion in [-1, 1]
        op = np.random.rand(ops)
        for i in range(ops):
            op[i] = op[i] * 2 - 1
        G.add_node(n , opinion = op)
        node_attribute[n] = op

    # Create homophilic distance ### faster to do it outside loop ###
    dist = defaultdict(float) #distance between nodes

    # Distance between nodes is the euclidean distance mapped to [0, 100]
    # Euclidean distance's value is in [0, sqrt(4 * ops)]
    max_distance = math.sqrt(4 * ops)
    for n1 in range(N):
        n1_attr = node_attribute[n1]
        for n2 in range(N):
            n2_attr = node_attribute[n2]
            euclidean_distance = np.linalg.norm(n1_attr - n2_attr)
            # Distance is mapped in [0, 100] for generation's sake
            dist[(n1,n2)] = euclidean_distance / max_distance * 100

    target_list = list(range(m))
    # Start with m nodes
    source = m 

    while source < N:
        targets = _pick_targets(G, source, target_list, dist, m, alpha=alpha ,beta=beta)
        if targets != set(): # If the node does find the neighbor
            G.add_edges_from(zip([source] * m, targets))
        target_list.append(source)  # Target list is updated with all the nodes in the graph 
        source += 1
    return G



def _pick_targets(G,source,target_list,dist,m ,alpha, beta):
    # First compute the target_prob which is related to the degree
    target_prob_dict = {}
    for target in target_list:
        pow_dist =  (dist[(source, target)] + 1) ** alpha
        target_prob = (1 / pow_dist) * ((G.degree(target) + 0.00001) ** beta) #Formula to compute targer prob, >>Degree better chance
        target_prob_dict[target] = target_prob
        
    prob_sum = sum(target_prob_dict.values())
    targets = set()
    target_list_copy = target_list.copy()
    count_looking = 0
    if prob_sum == 0:
        return targets # It returns an empty set

    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # If node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum:  
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets



'''
    Create_graph creates a graph using the method MY_homophilic_ba_graph,
    then it adds the beta used for BEBA and ABEBA models.

Parameters
----------
    n_ag : {int}
        The number of nodes in the graph.
    ops : {int}
        the number of opinions for each node.
    beba_beta : {list of float}
        The list with nodes' beta values.
    avg_friend: {}
        The number of average neighbors for each node in the graph.
    hp_alpha : {float}
        MY_homophilic_graph's homophily parameter.
    hp_beta : {float}
        MY_homophilic_graph's preferential attachment parameter.

Returns
-------
    G : {networkx.Graph}
        Returns the graph obtained.
'''

def create_graph(n_ag, ops=1,  beba_beta=[1] , avg_friend=3, hp_alpha=2, hp_beta=1):
  # Checks on beba_beta length
  if len(beba_beta) != 1 and len(beba_beta) != n_ag:
    print("WARNING: beba_beta length is not valid. It must be 1 or nodes' number. Default value will be used")
    beba_beta = [1] * n_ag

  if len(beba_beta) == 1:
    beba_beta = [beba_beta[0]] * n_ag

  # Calls MY_homophilic_ba_graph
  G = MY_homophilic_ba_graph(n_ag, avg_friend, ops, hp_alpha, hp_beta)

  # Setting initial attributes for the graph
  node_beba_beta_dict = dict(zip(G.nodes(), beba_beta))
  nx.set_node_attributes(G, node_beba_beta_dict, 'beba_beta')
  node_feed = dict(zip(G.nodes(), [[[] for i in range(ops)] for j in range(n_ag)]))
  nx.set_node_attributes(G, node_feed, 'feed')
  node_feed_history = dict(zip(G.nodes(), [[[] for i in range(ops)] for j in range(n_ag)]))
  nx.set_node_attributes(G, node_feed_history, 'feed_history')
  to_estim = dict(zip(G.nodes(), [[[] for i in range(ops)] for j in range(n_ag)]))
  nx.set_node_attributes(G, to_estim, 'to_estimate')
  estimated = dict(zip(G.nodes(), [[0.0] * ops for j in range(n_ag)]))
  nx.set_node_attributes(G, estimated, 'estimated_opinion')
  posteri_opinion = dict(zip(G.nodes(), [[0.0 for i in range(ops)] for j in range(n_ag)]))
  posteri_error = dict(zip(G.nodes(), [[1.0 for i in range(ops)] for j in range(n_ag)]))
  nx.set_node_attributes(G, posteri_opinion, 'posteri_opinion')
  nx.set_node_attributes(G, posteri_error, 'posteri_error')
  sat_dict = {node : {} for node in G.nodes()}
  nx.set_node_attributes(G, sat_dict, 'feed_satisfaction')
  return G