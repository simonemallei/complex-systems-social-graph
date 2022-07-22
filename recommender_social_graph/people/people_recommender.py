import networkx as nx
import numpy as np
from abeba_methods import compute_activation, compute_post

class RecommendedFriendNotFound(Exception):
    """Raised when the People Recommender fails to recommend a new friend to a given node"""
    pass

class SimulateEpochPeopleRecommenderError(Exception):
    """Raised when an error occurred in the simulate_epoch_people_recommender method"""
    pass

'''
x_hop_neighbors is called recursively and returns the nodes after the desired hops.
Since it receives an undirected graph as input, it excludes the nodes already visited 
to prevent "going back" in the exploration.

Parameters
----------
  G : {networkx.Graph}
      The graph containing the social network.
  nodes : {list of object}
      The list containing nodes' IDs on which to search for nodes at {x_hop}
  x_hop : {int}     
      Number of "hops" to do before stopping and returning found nodes
 previous_nodes : {list of ints}
      Contains the id of the nodes to be discarded to avoid "going back" 
      in the exploration of the graph and to return nodes that are not at 
      the correct number of hops compared to the starting node

Returns
-------
  all_new_friends_list_unique : {list of ints}
      the list the list of node ids that are exactly at x hops according to the starting nodes
'''
def x_hop_neighbors(G, nodes, x_hop, previous_nodes = None):
    all_friends_list = []
    if previous_nodes is None:
        previous_nodes = nodes
    for node in nodes:
        friends = list(nx.neighbors(G, node))
        all_friends_list += friends
    
    all_new_friends_list = [x for x in all_friends_list if x not in previous_nodes]
    all_new_friends_list_unique = list(dict.fromkeys(all_new_friends_list))
    if x_hop == 1:
        return all_new_friends_list_unique
    else:
        previous_nodes += friends
        return x_hop_neighbors(G, all_new_friends_list_unique, x_hop-1, previous_nodes)


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
        recommended_friend = None
        neigs = list(nx.neighbors(G, node_id))
        not_friends = [x for x in all_nodes if x not in neigs]
        not_friends.remove(node_id)
        if strategy == "random":
            nx.set_node_attributes(G, {node_id: not_friends}, 'people_recommended')
            # recommending a random node not already friend as a new friend. 
            recommended_friend = np.random.choice(not_friends, size=1, replace=False)
        elif strategy == 'opinion_diversity':
            distances_dict = {}
            opinions = nx.get_node_attributes(G, 'opinion')
            opinion_subject = opinions[node_id]
            for key in opinions:
                if key in not_friends:
                    abs_distance = abs(opinion_subject - opinions[key])
                    distances_dict[key] = abs_distance

            # ordered from the largest value to the smallest, therefore from the most discordant to the least discordant node
            distances_dict_ordered = {k: v for k, v in sorted(distances_dict.items(), key=lambda item: item[1], reverse=True)}
            # the choice is made between the first 4 keys of the dictionary
            short_list = list(distances_dict_ordered.keys())[0:4]
            nx.set_node_attributes(G, {node_id: short_list}, 'people_recommended')
            recommended_friend = np.random.choice(short_list, size=1, replace=False)
        # This strategy still doesn't work well
        # I need to understand what kind of nodes are accepted and then returned by the x_hop_neighbors method
        elif strategy == 'topology_based':
            overlapping_dict = {}
            x_hop_neighbors_list = x_hop_neighbors(G, [node_id], 3)
            x_hop_not_friends = [x for x in x_hop_neighbors_list if x in not_friends]
            for not_friend in x_hop_not_friends:
                not_friend_neigs = list(nx.neighbors(G, not_friend))
                # the number of mutual friendly nodes is given by the length of the intersection between the two friend lists
                # note that neither of the two lists can have duplicates
                number_overlapping_friends = len(set(not_friend_neigs) & set(neigs))
                overlapping_dict[not_friend] = number_overlapping_friends

            # ordered from the largest value to the smallest, therefore from the most overlapping friends number to the least overlapping friends number node 
            overlapping_dict_ordered = {k: v for k, v in sorted(overlapping_dict.items(), key=lambda item: item[1], reverse=True)}
            # the choice is made between the first 4 keys of the dictionary
            short_list = list(overlapping_dict_ordered.keys())[0:4]
            nx.set_node_attributes(G, {node_id: short_list}, 'people_recommended')
            recommended_friend = np.random.choice(short_list, size=1, replace=False)

        if recommended_friend is None:
            raise RecommendedFriendNotFound
        else:
            # note that recommended_friend is a numpy array with 1 element
            G.add_edge(node_id, recommended_friend[0])
            # deleting a random edge to prevent fully connected graphs
            discarded_friend = np.random.choice(neigs, size=1, replace=False)
            # note that discarded_friend is a numpy array with 1 element
            G.remove_edge(node_id, discarded_friend[0])
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
  try:
    G = people_recommender(G, post_nodes)
  except RecommendedFriendNotFound:
    print("the People Recommender failed to recommend a new friend to a given node")
    raise SimulateEpochPeopleRecommenderError

  return G
