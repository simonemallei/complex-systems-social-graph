import networkx as nx
import numpy as np
from collections import defaultdict
import random

'''
compute_engagement computes the engagement of a node with the given
opinion, beta and feed.

Parameters
----------
    node_opinion : {float}
        The opinion of the node we are considering.
    node_beta : {float}
        The ABEBA model's beta of the node.
    node_feeds : {list of float}
        The given node's feed.

Returns
-------
    engagement : {float}
        The engagement value.
'''
def compute_engagement(node_opinion, node_beta, node_feeds):
    engagement = 1
    if len(node_feeds) != 0:
        exp_difference = [np.exp(abs(feed_op - node_opinion)) - 1 for feed_op in node_feeds]
        engagement = (1 + node_beta * np.sum(exp_difference) / len(node_feeds))
    return engagement

'''
compute_activation performs the activation of the nodes contained in {nodes}.
The activation by a node is computed by updating his own opinion based
on his feed and following the ABEBA (Adapted BEBA) model.
The activated nodes' feed is then cleared since they no longer 
update their opinion based on that content.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    nodes : {list of object}
        The list containing activated nodes' IDs (dictionary keys).

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def compute_activation(G, nodes):
    opinions = nx.get_node_attributes(G, 'opinion')
    all_feeds = nx.get_node_attributes(G, 'feed')
    beba_beta_list = nx.get_node_attributes(G, 'beba_beta')
    prob_post = {}
    base_prob_post = nx.get_node_attributes(G, 'base_prob_post')

    # Removing old engagement values
    engagement_dict = nx.get_node_attributes(G, 'engagement')
    for key in engagement_dict.keys():
        del G.nodes[key]['engagement']
    engagement_dict = {}

    # Activating update of each node
    for curr_node in nodes:
        node_feeds = all_feeds.get(curr_node, [])

        # Computing weight w(i, i)
        weight_noose = beba_beta_list[curr_node] * opinions[curr_node] * opinions[curr_node] + 1

        # Computing engagement and using it as a coefficient of posting's probability
        engagement = compute_engagement(opinions[curr_node], beba_beta_list[curr_node], node_feeds)
        engagement_dict[curr_node] = engagement

        prob_post[curr_node] = min(1.0, base_prob_post[curr_node] * engagement)

        # Computing new opinion of curr_node
        op_num = weight_noose * opinions[curr_node]
        op_den = weight_noose
        for feed in node_feeds:
            # Computing weights w(i, j) where i == curr_node and y(j) == feed
            weight = beba_beta_list[curr_node] * feed * opinions[curr_node] + 1
            op_num += weight * feed
            op_den += weight

        # If the denominator is < 0, the opinion gets polarized and 
        # the value is set to sgn(opinions[curr_node])
        if op_den <= 0:
            opinions[curr_node] = opinions[curr_node] / abs(opinions[curr_node])
        else:
            opinions[curr_node] = op_num / op_den
      
        # Opinions are capped within [-1, 1] 
        if opinions[curr_node] < -1:
            opinions[curr_node] = -1
        if opinions[curr_node] > 1:
            opinions[curr_node] = 1
        all_feeds[curr_node] = []
      
  # Updating feed and opinion attributes
    nx.set_node_attributes(G, all_feeds, 'feed')
    nx.set_node_attributes(G, opinions, 'opinion')
    nx.set_node_attributes(G, prob_post, 'prob_post')
    nx.set_node_attributes(G, engagement_dict, 'engagement')

    return G

'''
compute_post performs the posting phase of the nodes contained in {nodes}.
To post an opinion, the node sends to his neighbourhood's feed his 
opinion with some noise (noise is defined as a normal distribution 
with mean = 0 and std = {epsilon}).
By sending to his neighbourhood's feed we mean updating his neighbourhood's
feed by adding his opinion (with noise) to it. 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    nodes : {list of object}
        The list containing activated nodes' IDs (dictionary keys).
    epsilon : {float}
        The Gaussian noise's standard deviation.

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def compute_post(G, nodes, epsilon = 0.0):
    opinions = nx.get_node_attributes(G, 'opinion')
    estim = {}
    prob_post = nx.get_node_attributes(G, 'prob_post')
    posting_nodes_list = []
    for node_id in nodes:
        sample = random.random()
        # {node_id} node will post with probability
        # {prob_post[node_id]}
        if sample <= prob_post[node_id]:
            posting_nodes_list.append(node_id)
            # epsilon defines the standard deviation of the value generated
            rand_eps = np.random.normal(0, epsilon, 1) 
            noise_op = rand_eps[0] + opinions[node_id]
            # Bounded in the range [-1, 1]
            noise_op = max(noise_op, -1)
            noise_op = min(noise_op, 1)

            post = [noise_op] # a list with one value
            past_feed = nx.get_node_attributes(G, 'feed') #get all user feed

            # Spread Opinion
            all_neig = list(nx.neighbors(G, node_id))   #get all neighbours ID

            # We have to estimate again the {node_id} opinion, since it has 
            # posted a new content
            estim[node_id] = noise_op

            post_to_be_added = {neig : list(post) for neig in all_neig}

            post_post_to_be_added = {key: past_feed[key] + value 
                                        if key in [*past_feed]
                                        else value
                                        for key, value in post_to_be_added.items()}
              
            # Debug print
            #print('POST ',  post_post_to_be_added)
            nx.set_node_attributes(G, post_post_to_be_added , name='feed')
    nx.set_node_attributes(G, estim, name='to_estimate')
    
    return G, posting_nodes_list

'''
simulate_epoch simulates an epoch. It randomly activates a 
subset ({rate_updating_nodes * len(G.nodes())}) of graph's nodes and they will
update their opinion depending on their feed.
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

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def simulate_epoch(G, rate_updating_nodes, epsilon = 0.0):
    # Sampling randomly the activating nodes
    updating_nodes = int(rate_updating_nodes * len(G.nodes()))
    act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)
    # Debug print
    #print(f"Activated nodes (consuming their feed): {act_nodes}")

    # Executing activation phase: activated nodes will consume their feed
    G = compute_activation(G, act_nodes)

    # Executing posting phase: activated nodes will post in their neighbours' feed
    G = compute_post(G, act_nodes, epsilon)
    return G
