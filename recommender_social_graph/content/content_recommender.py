import networkx as nx
import numpy as np
from collections import defaultdict
import random
from abeba_methods import compute_activation, compute_post
'''
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
  strategy : {"random", "normal", "similar", "unsimilar"} default: "random"
      The string that defines the strategy used by the recommender system.
  strat_param : {dictionary}
      The dictionary containing the parameters value used by the recommender, based
      on {strategy} value.
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
def content_recommender(G, act_nodes, strategy="random", strat_param={}):
  feed = nx.get_node_attributes(G, 'feed')
  opinions = nx.get_node_attributes(G, 'opinion')
  new_feed = dict()
  for node_id in act_nodes:
    if strategy == "random":
      # Generating random recommended content in the range [-1, 1]
      recommend_cont = np.random.uniform(-1, 1) 
      post = [recommend_cont] # a list with one value
      new_feed[node_id] = feed.get(node_id, []) + post
    elif strategy == "normal":
      normal_mean = strat_param.get('normal_mean', 0.0)
      normal_std = strat_param.get('normal_std', 0.1)
      # Generating recommended content using a normal distribution with
      # the following parameters: mean = {normal_mean}, std = {normal_std}
      recommend_cont = np.random.normal(normal_mean, normal_std)
      recommend_cont = min(1, max(-1, recommend_cont))
      post = [recommend_cont] # a list with one value
      new_feed[node_id] = feed.get(node_id, []) + post
    elif strategy == "similar":
      similar_thresh = strat_param.get('similar_thresh', 0.5)
      curr_op = opinions[node_id]
      # Deleting content that is too far away from the node's opinion (measuring the distance
      # as the absolute difference between the content's opinion and the node's one) 
      prev_feed = feed.get(node_id, [])
      new_feed[node_id] = [post for post in prev_feed if abs(post - curr_op) <= similar_thresh]
    elif strategy == "unsimilar":
      unsimilar_thresh = strat_param.get('unsimilar_thresh', 0.3)
      curr_op = opinions[node_id]
      # Deleting content that is too close from the node's opinion (measuring the distance
      # as the absolute difference between the content's opinion and the node's one) 
      prev_feed = feed.get(node_id, [])
      new_feed[node_id] = [post for post in prev_feed if abs(post - curr_op) >= unsimilar_thresh]
  # Updating feed with recommended content  
  nx.set_node_attributes(G, new_feed , name='feed')
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
def monitor_feed(G, act_nodes):
  feed = nx.get_node_attributes(G, 'feed')
  feed_history = nx.get_node_attributes(G, 'feed_history')
  for node_id in act_nodes:
    # Updating feed history for each activated nodes
    curr_history = feed_history.get(node_id, [])
    curr_feed = feed.get(node_id, [])
    feed_history[node_id] = curr_history + curr_feed
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
def simulate_epoch_content_recommender(G, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0, strategy = "random", strat_param = {}):
  # Sampling randomly the activating nodes
  updating_nodes = int(percent_updating_nodes * len(G.nodes()) / 100)
  act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)
  # Debug print
  #print(f"Activated nodes (consuming their feed): {act_nodes}")

  # Executing content recommender system on activated nodes
  G = content_recommender(G, act_nodes, strategy, strat_param)
  # Monitoring feeds that are going to be cleared 
  G = monitor_feed(G, act_nodes)
  # Executing activation phase: activated nodes will consume their feed
  G = compute_activation(G, act_nodes)

  # Sampling randomly the posting nodes from activating nodes' list
  posting_nodes = int(percent_posting_nodes * len(act_nodes) / 100)
  post_nodes = np.random.choice(act_nodes,size=posting_nodes, replace = False)
  # Debug print
  #print(f"Posting nodes: {post_nodes}")

  # Executing posting phase: activated nodes will post in their neighbours' feed
  G = compute_post(G, post_nodes, epsilon)
  return G
