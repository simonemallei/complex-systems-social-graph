import networkx as nx
import numpy as np

from graph_creation import create_graph


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
    ops: {int}
        The number of opinion of each user
Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''

def compute_activation(G, nodes, ops):
    opinions = nx.get_node_attributes(G, 'opinion')
    all_feeds = nx.get_node_attributes(G, 'feed')
    beba_beta_list = nx.get_node_attributes(G, 'beba_beta')

    # Activating update of each node
    for curr_node in nodes:
        node_feeds = all_feeds.get(curr_node, [])

    # Computing weight w(i, i)
    # w[i][j] = beta[i] * dot_product(opinion[i], opinion[j]) + 1
    weight_noose = beba_beta_list[curr_node] * np.dot(opinions[curr_node], opinions[curr_node]) / ops + 1

    # Computing new opinion of curr_node
    op_num = weight_noose * opinions[curr_node]
    op_den = weight_noose
    for feed in node_feeds:
        # Computing weights w(i, j) where i == curr_node and y(j) == feed
        weight = beba_beta_list[curr_node] * np.dot(feed, opinions[curr_node]) / ops + 1
        op_num += weight * feed
        op_den += weight

    # If the denominator is < 0, the opinion gets polarized and 
    # the value is set to sgn(opinions[curr_node])
    if op_den <= 0:
        for current_op in range(ops):
            opinions[curr_node][current_op] = opinions[curr_node][current_op] / abs(opinions[curr_node][current_op])
    else:
        opinions[curr_node] = op_num / op_den

    # Opinions are capped within [-1, 1] 
    for current_op in range(ops):
        if opinions[curr_node][current_op] < -1:
            opinions[curr_node][current_op] = -1
        if opinions[curr_node][current_op] > 1:
            opinions[curr_node][current_op] = 1
    all_feeds[curr_node] = []

    # Updating feed and opinion attributes
    nx.set_node_attributes(G, all_feeds, 'feed')
    nx.set_node_attributes(G, opinions, 'opinion')
    return G



"""
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
        The list containing posting nodes' IDs (dictionary keys).
    ops : {int}
        The number of opinions of each node
    epsilon : {float}
        The Gaussian noise's standard deviation.

Returns
-------
    G : {networkx.Graph}
      The updated graph.
"""

def compute_post(G, nodes, ops, epsilon = 0.0):
    opinions = nx.get_node_attributes(G, 'opinion')
    for node_id in nodes:
        new_opinion = opinions[node_id]
        for op in range(ops):
            rand_eps = np.random.normal(0, epsilon, 1)
            noise_op = rand_eps[0] + opinions[node_id][op]
            noise_op = min(noise_op, 1)
            noise_op = max(noise_op, -1)
            new_opinion[op] = noise_op
        post = [new_opinion]
        past_feed = nx.get_node_attributes(G, 'feed')

        # Spread Opinion
        all_neig = list(nx.neighbors(G, node_id))   #get all neighbours ID

        post_to_be_added = dict(zip(all_neig,
                                   [list(post) for _ in range(len(all_neig))] ))
        post_post_to_be_added = {key: past_feed[key] + value 
                              if key in [*past_feed]
                              else value
                              for key, value in post_to_be_added.items()}

        nx.set_node_attributes(G, post_post_to_be_added , name='feed')
    return G



'''
simulate_epoch_updated simulates an epoch. It randomly activates a 
percentage ({percent_updating_nodes}) of graph's nodes and they will
update their opinion base on their feed.
Afterwards, a percentage equal to {percentage_posting_nodes} of the activated
vertices (always sampled randomly) will also be posting nodes, updating 
their neighbours' feed with the content. 
The opinion shared by the posting nodes has a noise related
to the parameter {epsilon}.

Parameters
    ----------
    G : {networkx.Graph}
        The graph containing the social network.
    ops : {int}
        The number of opinions of each node
    percent_updating_nodes : {int}
        The percentage of the nodes that will be activated.
    percent_posting_nodes : {int}
        The percentage of the activated nodes that will be posting nodes as well.
    epsilon : {float}
        The Gaussian noise's standard deviation in the posting phase.

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''

def simulate_epoch_updated(G, ops, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0):
    # Sampling randomly the activating nodes
    updating_nodes = int(percent_updating_nodes * len(G.nodes()) / 100)
    act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)

    # Executing activation phase: activated nodes will consume their feed
    G = compute_activation(G, act_nodes, ops)

    # Sampling randomly the posting nodes from activating nodes' list
    posting_nodes = int(percent_posting_nodes * len(act_nodes) / 100)
    post_nodes = np.random.choice(act_nodes,size=posting_nodes, replace = False)

    # Executing posting phase: activated nodes will post in their neighbours' feed
    G = compute_post(G, post_nodes, ops, epsilon)
    return G



'''
apply_initial_feed creates an initial feed for each node.
A vertex feed will contain {n_post} posts with an opinion that is
the node's one with some noise (the noise is generated with a normal
distribution with mean = 0 and a std = {epsilon}).

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    ops : {int}
        The number of opinions of each node
    n_post : {int}
        The number of posts created for each node in the initial feed.
    epsilon : {float}
        The Gaussian noise's standard deviation.

Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''

def apply_initial_feed(G, ops, n_post = 10, epsilon = 0.1):
  # Casting all the numpy arrays as built-in lists 
  initial_feed_dict = dict()
  opinions = nx.get_node_attributes(G, 'opinion')

  for curr_node in G.nodes():
    # Sampling {n_post} elements from a normal distribution with
    # - mean = 0.0
    # - std = epsilon
    # This values are added with the original opinion in order to have 
    # a feed that has similar values with the starting opinion
    feed = [np.random.normal(0, epsilon, ops) + opinions[curr_node] for i in range(n_post)]
    for i in range(n_post):
        for j in range(ops):
            feed[i][j] = max(-1, min(1, feed[i][j]))
    initial_feed_dict[curr_node] = list(feed)
  
  # Setting these values as feed in the graph
  nx.set_node_attributes(G, initial_feed_dict, 'feed')

  return G



g = create_graph(5, ops=3,  beba_beta=[1] , avg_friend=3, hp_alpha=2, hp_beta=1)
compute_activation(g, {1, 2}, 3)
compute_post(g, {1, 2}, 3, 0.1)