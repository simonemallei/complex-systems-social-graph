from platform import node
from turtle import distance
import networkx as nx
import numpy as np
from collections import defaultdict
import random
# Use this for notebook
# from multi_dimensional.abeba_methods import compute_activation, compute_post
#Use this for test.py
from abeba_methods import *
from content.content_recommender import *
import math
from tabulate import tabulate
from test import print_graph

def people_recommender(G, ops, nodes, strategy="random"):
    all_nodes = list(G.nodes)
    for node_id in nodes:
        recommended_friend = []
        neigs = list(nx.neighbors(G, node_id))
        not_friends = [x for x in all_nodes if x not in neigs]
        not_friends.remove(node_id)
        if strategy == "random":
            nx.set_node_attributes(G, {node_id: not_friends}, 'people_recommended')
            # recommending a random node not already friend as a new friend.
            recommended_friend = []
            if len(not_friends) != 0: 
                recommended_friend = np.random.choice(not_friends, size=1, replace=False)
        elif strategy == 'opinion_diversity':
            distances_dict = {}
            opinions = nx.get_node_attributes(G, 'estimated_opinion')
            opinion_subject = opinions[node_id]
            for key in opinions:
                if key in not_friends:
                    distances_dict[key] = 0.0
                    for op in range(ops):
                        distances_dict[key] += (opinion_subject[op] - opinions[key][op]) ** 2
            # ordered from the largest value to the smallest, therefore from the most discordant to the least discordant node
            distances_dict_ordered = {k: v for k, v in sorted(distances_dict.items(), key=lambda item: item[1], reverse=True)}
            # the choice is made between the first 4 keys of the dictionary
            sz = len(distances_dict)
            short_list = list(distances_dict_ordered.keys())[0:min(sz, 4)]
            nx.set_node_attributes(G, {node_id: short_list}, 'people_recommended')
            recommended_friend = np.random.choice(short_list, size=1, replace=False)
        elif strategy == 'topology_based':
            overlapping_dict = {}
            # BFS 
            visited, queue = [], []
            visited.append(node_id)
            queue.append((node_id, 0))
            while queue:
                # next pair (node, distance)
                next = queue.pop(0)
                # reached max distance
                if next[1] > 3:
                    break
                next_neigs = list(nx.neighbors(G, next[0]))
                for neig in next_neigs:
                    if neig not in visited:
                        visited.append(neig)
                        queue.append((neig, next[1] + 1))
            visited = [x for x in visited if x not in neigs]
            visited.remove(node_id)
            for not_friend in visited:
                not_friend_neigs = list(nx.neighbors(G, not_friend))
                number_overlapping_friends = len(set(not_friend_neigs) & set(neigs))
                overlapping_dict[not_friend] = number_overlapping_friends
            # ordered from the largest value to the smallest, therefore from the most overlapping friends number to the least overlapping friends number node 
            overlapping_dict_ordered = {k: v for k, v in sorted(overlapping_dict.items(), key=lambda item: item[1], reverse=True)}
            # the choice is made between the first 4 keys of the dictionary
            short_list = list(overlapping_dict_ordered.keys())[0:min(4, len(overlapping_dict_ordered))]
            nx.set_node_attributes(G, {node_id: short_list}, 'people_recommended')
            recommended_friend = []
            if (len(short_list) > 0):
                recommended_friend = np.random.choice(short_list, size=1, replace=False)

        # note that recommended_friend is a numpy array with at most 1 element
        if len(recommended_friend) > 0:
            G.add_edge(node_id, recommended_friend[0])
            # deleting a random edge to prevent fully connected graphs
            discarded_friend = []
            if len(neigs) > 0:
                discarded_friend = np.random.choice(neigs, size=1, replace=False)
                G.remove_edge(node_id, discarded_friend[0])
            """
            print('Node:\t' + str(node_id))
            print('New Friend:\t' + str(recommended_friend))
            print('Old Friend:\t' + str(discarded_friend))
            """
    return G



def simulate_epoch_content_people_recommender(G, ops, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0, 
      strategy = "random", strat_param = {}, estim_strategy='base', estim_strat_param={}, people_strategy='random'):
  # Sampling randomly the activating nodes
  updating_nodes = int(percent_updating_nodes * len(G.nodes()) / 100)
  act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)

  # Executing content recommender system on activated nodes
  G = content_recommender(G, ops, act_nodes, strategy, strat_param)
  # Monitoring feeds that are going to be cleared 
  G = monitor_feed(G, ops, act_nodes)
  # Executing activation phase: activated nodes will consume their feed
  G = compute_activation(G, act_nodes, ops)

  # Sampling randomly the posting nodes from activating nodes' list
  posting_nodes = int(percent_posting_nodes * len(act_nodes) / 100)
  post_nodes = np.random.choice(act_nodes,size=posting_nodes, replace = False)

  # Executing posting phase: activated nodes will post in their neighbours' feed
  G = compute_post(G, post_nodes, ops, epsilon)

  # Updating estimated opinion 
  G = upd_estim(G, ops, strategy=estim_strategy, strat_param=estim_strat_param)

  #Calling people recommender
  G = people_recommender(G, ops, post_nodes, people_strategy)
  return G
