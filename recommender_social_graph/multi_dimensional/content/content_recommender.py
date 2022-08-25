from platform import node
import networkx as nx
import numpy as np
from collections import defaultdict
import random
# Use this for notebook
# from multi_dimensional.abeba_methods import compute_activation, compute_post
#Use this for test.py
from abeba_methods import compute_activation, compute_post
import math
from tabulate import tabulate

"""
content_recommender performs the recommender system by content.
It performs the recommendation routine on the nodes contained 
in {act_nodes}.
The recommender routine depends on which strategy is chosen 
(by default, the method use the "random" one). 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    act_nodes : {list of object}
        The list containing activated nodes' IDs (dictionary keys).
    strategy : {"random", "normal", "nudge", "nudge_opt", "similar", "unsimilar"} default: "random"
        The string that defines the strategy used by the recommender system.
    strat_param : {dictionary}
        The dictionary containing the parameters value used by the recommender, based
        on {strategy} value.
            - key: "n_post",
              value: number of posts added in activated node's feed by the recommender.
            - key: "normal_mean",
              value: mean of distribution of values produced by "normal" strategy.
            - key: "normal_std",
              value: standard dev. of distribution of values produced by "normal" strategy.
            - key: "nudge_goal",
              value: the "opinion goal" by the nudging content recommender.
            - key: "similar_thresh",
              value: threshold value used by "similar" strategy.
            - key: "unsimilar_thresh",
              value: threshold value used by "unsimilar" strategy.
        
Returns
-------
    G : {networkx.Graph}
        The updated graph.
"""

def content_recommender(G, ops, act_nodes, strategy="random", strat_param={}):
  feed = nx.get_node_attributes(G, 'feed')
  opinions = nx.get_node_attributes(G, 'opinion')
  beta = nx.get_node_attributes(G, 'beba_beta')
  new_feed = dict()
  for node_id in act_nodes:
    if strategy == "random":
      # Generating random recommended content in the range [-1, 1]
      # (n_post posts in the feed) 
      n_post = strat_param.get('n_post', 1)
      for i in range(n_post):
        # For each post a random dimension is chosen 
        op = random.randint(0, ops - 1)
        opinion = random.uniform(-1.0, 1.0)
        feed[node_id][op].append(opinion)
    elif strategy == "normal":
      normal_mean = strat_param.get('normal_mean', 0.0)
      normal_std = strat_param.get('normal_std', 0.1)
      # Generating recommended content using a normal distribution with
      # the following parameters: mean = {normal_mean}, std = {normal_std}
      # (n_post posts in the feed) 
      post = []
      n_post = strat_param.get('n_post', 1)
      for i in range(n_post):
        recommend_cont = np.random.normal(normal_mean, normal_std, ops)
        for op in range(ops):
          recommend_cont[op] = min(1, max(-1, recommend_cont[op]))
        post.append(recommend_cont)
      new_feed[node_id] = feed.get(node_id, []) + post
    elif strategy == 'nudge':
      # Generating recommended content using a normal distribution with
      # the following parameters: mean = {nudge_mean}, std = {nudge_std}
      # (n_post posts in the feed)
      curr_op = opinions[node_id]
      nudge_goal = strat_param.get('nudge_goal', np.zeros(ops))
      max_dist = math.sqrt(4 * ops)
      dist = np.linalg.norm(curr_op - nudge_goal) / max_dist
      nudge_std = (dist / 4 * (1 / (2 ** beta[node_id])))
      n_post = strat_param.get('n_post', 1)
      post = []
      for i in range(n_post):
        current = np.zeros(ops)
        for opinion in range(ops):
          current[opinion] = np.random.normal(nudge_goal[opinion], nudge_std)
          current[opinion] = min(1, max(-1, current[opinion]))
        post.append(current)
      new_feed[node_id] = feed.get(node_id, []) + post
    elif strategy == 'nudge_opt':
      curr_op = opinions[node_id]
      nudge_goal = strat_param.get('nudge_goal', np.zeros(ops))
      copy_goal = np.copy(nudge_goal)
      for opinion in range(ops):
        if curr_op[opinion] * copy_goal[opinion] < 0.0:
          copy_goal[opinion] = 0.0
      max_dist = math.sqrt(4 * ops)
      dist = np.linalg.norm(curr_op - copy_goal) / max_dist
      nudge_std = (dist / 4 * (1 / (2 ** beta[node_id])))
      n_post = strat_param.get('n_post', 1)
      post = []
      for i in range(n_post):
        current = np.zeros(ops)
        for opinion in range(ops):
          current[opinion] = np.random.normal(copy_goal[opinion], nudge_std)
          current[opinion] = min(1, max(-1, current[opinion]))
        post.append(current)
      new_feed[node_id] = feed.get(node_id, []) + post
    elif strategy == "similar":
      similar_thresh = strat_param.get('similar_thresh', 0.5)
      curr_op = opinions[node_id]
      # Deleting content that is too far away from the node's opinion (measuring the distance
      # as the absolute difference between the content's opinion and the node's one) 
      prev_feed = feed.get(node_id, [])
      max_distance = math.sqrt(4 * ops)
      new_feed[node_id] = [post for post in prev_feed if np.linalg.norm(curr_op - post) / max_distance <= similar_thresh]
    elif strategy == "unsimilar":
      unsimilar_thresh = strat_param.get('unsimilar_thresh', 0.3)
      curr_op = opinions[node_id]
      # Deleting content that is too close from the node's opinion (measuring the distance
      # as the absolute difference between the content's opinion and the node's one) 
      prev_feed = feed.get(node_id, [])
      max_distance = math.sqrt(4 * ops)
      new_feed[node_id] = [post for post in prev_feed if np.linalg.norm(curr_op - prev_feed) / max_distance >= unsimilar_thresh]
  # Updating feed with recommended content  
  nx.set_node_attributes(G, feed , name='feed')
  return G

'''
monitor_feed performs the monitoring of each node's feed.
It uses the 'feed_history' graph attribute in order
to register all the posts that are read by each node.

Parameters
----------
  G : {networkx.Graph}
      The graph containing the social network.
  act_nodes : {list of object}
      The list containing activated nodes' IDs (dictionary keys).
        
Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def monitor_feed(G, ops, act_nodes):
  feed = nx.get_node_attributes(G, 'feed')
  feed_history = nx.get_node_attributes(G, 'feed_history')
  for node_id in act_nodes:
    # Updating feed history for each activated nodes
    curr_history = feed_history.get(node_id, [[] for i in range(ops)])
    curr_feed = feed.get(node_id, [[] for i in range(ops)])
    for op in range(ops):
      result = curr_history[op] + curr_feed[op]
      feed_history[node_id][op] = curr_history[op] + curr_feed[op]
  # Updating the history in the graph
  nx.set_node_attributes(G, feed_history, name='feed_history')
  return G
  
'''
simulate_epoch_content_recommender simulates an epoch. It randomly activates a 
percentage ({percent_updating_nodes}) of graph's nodes: firstly the content
recommender will update their feed, then each node will update its opinion 
base on its feed.
Afterwards, a percentage equal to {percentage_posting_nodes} of the activated
vertices (always sampled randomly) will also be posting nodes, updating 
their neighbours' feed with the content. 
The opinion shared by the posting nodes has a noise related
to the parameter {epsilon}.

Parameters
----------
  G : {networkx.Graph}
      The graph containing the social network.
  percent_updating_nodes : {int}
      The percentage of the nodes that will be activated.
  percent_posting_nodes : {int}
      The percentage of the activated nodes that will be posting nodes as well.
  epsilon : {float}
      The Gaussian noise's standard deviation in the posting phase.
  strategy : {"random", "normal", "similar"} default: "random"
      The string that defines the strategy used by the recommender system.
  strat_param : {dictionary}
      The dictionary containing the parameters value based on the {strategy} value.
      - key: "normal_mean",
        value: mean of distribution of values produced by "normal" strategy.
      - key: "normal_std",
        value: standard dev. of distribution of values produced by "normal" strategy.
      - key: "similar_thresh",
        value: threshold value used by "similar" strategy.
      - key: "unsimilar_thresh",
        value: threshold value used by "unsimilar" strategy.

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def simulate_epoch_content_recommender(G, ops, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0, strategy = "random", strat_param = {}):
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
  return G
