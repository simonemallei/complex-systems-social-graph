import networkx as nx
import numpy as np
from collections import defaultdict
import random
from abeba_methods import compute_activation, compute_post
from estimation import upd_estim
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
    strategy : {"random", "normal", "nudge", "similar", "unsimilar"} default: "random"
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
'''
def content_recommender(G, act_nodes, strategy="random", strat_param={}):
    feed = nx.get_node_attributes(G, 'feed')
    opinions = nx.get_node_attributes(G, 'estimated_opinion')
    posteri_error = nx.get_node_attributes(G, 'posteri_error')
    new_feed = dict()
    for node_id in act_nodes:
        error_var = posteri_error.get(node_id, 1.0)
        if strategy == "random":
            # Generating random recommended content in the range [-1, 1]
            # (n_post posts in the feed) 
            n_post = strat_param.get('n_post', 1)
            post = [np.random.uniform(-1, 1) for _ in range(n_post)]
            #recommend_cont = np.random.uniform(-1, 1) 
            #post = [recommend_cont] # a list with one value
            new_feed[node_id] = feed.get(node_id, []) + post
        elif strategy == "normal":
            # Generating recommended content using a normal distribution with
            # the following parameters: mean = {normal_mean}, std = {normal_std}
            # (n_post posts in the feed) 
            normal_mean = strat_param.get('normal_mean', 0.0)
            normal_std = strat_param.get('normal_std', 0.1)
            n_post = strat_param.get('n_post', 1)
            
            post = [min(1, max(-1, np.random.normal(normal_mean, normal_std))) 
                    for _ in range(n_post)]
            new_feed[node_id] = feed.get(node_id, []) + post
        elif strategy == "nudge":
            # Generating recommended content using a normal distribution with
            # the following parameters: mean = {nudge_mean}, std = {nudge_std}
            # (n_post posts in the feed) 
            if (error_var < 7e-3):
                nudge_goal = strat_param.get('nudge_goal', 0.0)
                node_op = opinions.get(node_id, 0.0)
                nudge_mean = (nudge_goal + node_op) / 2
                nudge_std = abs(nudge_mean - node_op) / 8
                n_post = strat_param.get('n_post', 1)
                
                post = [min(1, max(-1, np.random.normal(nudge_mean, nudge_std))) 
                        for _ in range(n_post)]
                new_feed[node_id] = feed.get(node_id, []) + post
        elif strategy == "similar":
            similar_thresh = strat_param.get('similar_thresh', 0.5)
            if (error_var < 7e-3):
                curr_op = opinions.get(node_id, [])
                # Deleting content that is too far away from the node's opinion (measuring the distance
                # as the absolute difference between the content's opinion and the node's one)
                # if we haven't estimated yet its opinion, there are no posts removed from the feed
                if (curr_op == []):
                    new_feed[node_id] = feed.get(node_id, [])
                else:
                    prev_feed = feed.get(node_id, [])
                    new_feed[node_id] = [post for post in prev_feed if abs(post - curr_op) <= similar_thresh]
        elif strategy == "unsimilar":
            unsimilar_thresh = strat_param.get('unsimilar_thresh', 0.3)
            if (error_var < 7e-3):
                curr_op = opinions.get(node_id, [])
                # Deleting content that is too close from the node's opinion (measuring the distance
                # as the absolute difference between the content's opinion and the node's one) 
                # if we haven't estimated yet its opinion, there are no posts removed from the feed
                if (curr_op == []):
                    new_feed[node_id] = feed.get(node_id, [])
                else:
                    prev_feed = feed.get(node_id, [])
                    new_feed[node_id] = [post for post in prev_feed if abs(post - curr_op) >= unsimilar_thresh]
    # Updating feed with recommended content  
    nx.set_node_attributes(G, new_feed, name='feed')
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
    feed_length = nx.get_node_attributes(G, 'feed_length')
    for node_id in G.nodes():
        feed_length[node_id] = 0
    for node_id in act_nodes:
        # Updating feed history for each activated nodes
        curr_history = feed_history.get(node_id, [])
        curr_feed = feed.get(node_id, [])
        feed_length[node_id] = len(curr_feed)
        feed_history[node_id] = curr_history + curr_feed

    # Updating the history in the graph
    nx.set_node_attributes(G, feed_history, name='feed_history')
    nx.set_node_attributes(G, feed_length, name='feed_length')
    return G
  
'''
simulate_epoch_content_recommender simulates an epoch. It randomly activates a 
subset ({rate_updating_nodes * len(G.nodes())}) of graph's nodes: firstly the content
recommender will update their feed, then each node will update its opinion 
base on its feed.
Afterwards, each activated node will post their opinion in their feed with a 
a probability depending on each node (look compute_posting).
The opinion shared by the posting nodes has a noise related
to the parameter {epsilon}.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    rate_updating_nodes : {float}
        The rate of the nodes that will be activated.
    epsilon : {float}
        The Gaussian noise's standard deviation in the posting phase.
    strategy : {"random", "normal", "nudge", "nudge_opt", "similar", "unsimilar"} default: "random"
        The string that defines the strategy used by the recommender system.
    strat_param : {dictionary}
        The dictionary containing the parameters value based on the {strategy} value.
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
    estim_strategy : {"base", "kalman"} default: "base"
        The string that defines the estimation strategy used by the recommender system.
    estim_strat_param : {dictionary}
        The dictionary containing the parameters value used by the recommender, based
        on {estim_strategy} value.
            - key: "alpha",
              value: alpha coefficient used in "base" estimation strategy.
            - key: "variance",
              value: process variance of "kalman" estimation strategy.
            - key: "variance_measure",
              value: measure variance of "kalman" estimation strategy.

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def simulate_epoch_content_recommender(G, rate_updating_nodes, epsilon = 0.0, strategy = "random", strat_param = {},
                                      estim_strategy = "base", estim_strat_param = {}):
    # Sampling randomly the activating nodes
    updating_nodes = int(rate_updating_nodes * len(G.nodes()))
    act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)
    # Debug print
    #print(f"Activated nodes (consuming their feed): {act_nodes}")

    # Executing content recommender system on activated nodes
    G = content_recommender(G, act_nodes, strategy, strat_param)
    # Monitoring feeds that are going to be cleared 
    G = monitor_feed(G, act_nodes)
    # Executing activation phase: activated nodes will consume their feed
    G = compute_activation(G, act_nodes)

    # Executing posting phase: activated nodes will post in their neighbours' feed
    G = compute_post(G, act_nodes, epsilon)
    # Estimating opinion by the recommender
    G = upd_estim(G, strategy = estim_strategy, strat_param = estim_strat_param)
    return G
