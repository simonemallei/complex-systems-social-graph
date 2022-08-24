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
'''
def content_recommender(G, act_nodes, strategy="random", strat_param={}):
    feed = nx.get_node_attributes(G, 'feed')
    opinions = nx.get_node_attributes(G, 'estimated_opinion')
    beta = nx.get_node_attributes(G, 'beba_beta')
    new_feed = dict()
    for node_id in act_nodes:
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
            nudge_goal = strat_param.get('nudge_goal', 0.0)
            nudge_std = (abs(nudge_goal - opinions[node_id]) / 4 
                         * (1 / (2 ** beta[node_id])))
            n_post = strat_param.get('n_post', 1)
            
            post = [min(1, max(-1, np.random.normal(nudge_goal, nudge_std))) 
                    for _ in range(n_post)]
            new_feed[node_id] = feed.get(node_id, []) + post
        elif strategy == "nudge_opt":
            # Generating recommended content using a normal distribution with
            # the following parameters: mean = {nudge_mean}, std = {nudge_std}
            # (n_post posts in the feed) 
            nudge_goal = strat_param.get('nudge_goal', 0.0)
            if (opinions[node_id] * nudge_goal) < 0.0:
                nudge_goal = 0.0
            nudge_std = (abs(nudge_goal - opinions[node_id]) / 4 
                         * (1 / (2 ** beta[node_id])))
            n_post = strat_param.get('n_post', 1)
            
            post = [min(1, max(-1, np.random.normal(nudge_goal, nudge_std))) 
                    for _ in range(n_post)]
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

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def simulate_epoch_content_recommender(G, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0, strategy = "random", strat_param = {},
                                      estim_strategy = "base", estim_strat_param = {}):
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
    # Estimating opinion by the recommender
    G = upd_estim(G, strategy = estim_strategy, strat_param = estim_strat_param)
    return G


def upd_estim(G, strategy = "base", strat_param = {}):
    # New opinions to estimate (it contains the last content the nodes has posted
    # in the last epoch)
    to_estim = nx.get_node_attributes(G, name='to_estimate')
    # Already estimated opinions by the recommender
    estimated = nx.get_node_attributes(G, name='estimated_opinion')
    if strategy == "base":
        alpha = strat_param.get('alpha', 0.75)
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [])
            estim_op = estimated.get(node_id, 0.0)
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            if not(last_post == []):
                estimated[node_id] = estim_op * alpha + last_post * (1 - alpha)
            to_estim[node_id] = []
            
    elif strategy == "kalman":
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [])
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            if not(last_post == []):
                variance = strat_param.get('variance', 1e-5) # process variance
                R = strat_param.get('variance_measure', 0.1 ** 2) # estimate of measurement variance, change to see effect
                posteri_opinion = nx.get_node_attributes(G, name='posteri_opinion')
                posteri_error = nx.get_node_attributes(G, name='posteri_error')
                # Opinion a posteri (represents the last estimation)
                op_posteri = posteri_opinion.get(node_id, 0.0)
                # Error a posteri (represents the last error value)
                P_posteri = posteri_error.get(node_id, 1.0)


                # Using last posteri values (adding variance to error) as priori in the new epoch
                op_priori = op_posteri
                P_priori = P_posteri + variance
    
                # measurement update
                K = P_priori/(P_priori + R)
                # Compute new opinion and error posteri
                op_posteri = op_priori + K * (last_post - op_priori)
                P_posteri = (1 - K) * P_priori

                # Updating values obtained
                estimated[node_id] = op_posteri
                posteri_opinion[node_id] = op_posteri
                posteri_error[node_id] = P_posteri
                # Updating estimates
                nx.set_node_attributes(G, op_posteri, name='posteri_opinion')
                nx.set_node_attributes(G, P_posteri, name='posteri_error')
            
            to_estim[node_id] = []
    # Updating estimated opinions
    nx.set_node_attributes(G, to_estim, name='to_estimate')  
    nx.set_node_attributes(G, estimated, name='estimated_opinion')
    return G
