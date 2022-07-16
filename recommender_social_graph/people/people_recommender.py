import networkx as nx
import numpy as np
from abeba_methods import compute_activation, compute_post

'''
people_recommender performs a people recommender system.
It performs the recommendation routine on the nodes contained 
in {act_nodes}.
The recommender routine depends on which strategy is chosen 
(by default, the method use the "random" one). 

Parameters
----------
  G : {networkx.Graph}
      The graph containing the social network.
  nodes : {list of object}
      The list containing nodes' IDs (dictionary keys) on which to run the people recommender
  strategy : {"random"} default: "random"
      The string that defines the strategy used by the recommender system.

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def people_recommender(G, nodes, strategy="random"):
    all_nodes = list(G.nodes)
    for node_id in nodes:
        neigs = list(nx.neighbors(G, node_id))
        recommended_friends = [x for x in all_nodes if x not in neigs]
        recommended_friends.remove(node_id)
        if strategy == "random":
            # recommending a random node not already friend as a new friend
            recommended_friend = np.random.choice(range(len(recommended_friends)), size=1, replace=False)
            G.add_edge(node_id, recommended_friend)
            # deleting a random edge to prevent fully connected graphs
            discarded_friend = np.random.choice(range(len(neigs)), size=1, replace=False)
            G.remove_edge(node_id, discarded_friend)

    return G

'''
simulate_epoch_people_recommender simulates an epoch. It randomly activates a 
percentage ({percent_updating_nodes}) of graph's nodes and so each node will 
update its opinion base on its feed.
Afterwards, a percentage equal to {percentage_posting_nodes} of the activated
vertices (always sampled randomly) will also be posting nodes, updating 
their neighbours' feed with the content. The opinion shared by the posting nodes 
has a noise related to the parameter {epsilon}.
Then the people recommender is run on the posting_nodes.

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

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def simulate_epoch_people_recommender(G, percent_updating_nodes, percent_posting_nodes, epsilon = 0.0):
  # Sampling randomly the activating nodes
  updating_nodes = int(percent_updating_nodes * len(G.nodes()) / 100)
  act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)
  # Debug print
  #print(f"Activated nodes (consuming their feed): {act_nodes}")

  # Executing activation phase: activated nodes will consume their feed
  G = compute_activation(G, act_nodes)

  # Sampling randomly the posting nodes from activating nodes' list
  posting_nodes = int(percent_posting_nodes * len(act_nodes) / 100)
  post_nodes = np.random.choice(act_nodes,size=posting_nodes, replace = False)
  # Debug print
  #print(f"Posting nodes: {post_nodes}")

  # Executing posting phase: activated nodes will post in their neighbours' feed
  G = compute_post(G, post_nodes, epsilon)
  # Executing content recommender system on activated nodes
  G = people_recommender(G, post_nodes)

  return G
