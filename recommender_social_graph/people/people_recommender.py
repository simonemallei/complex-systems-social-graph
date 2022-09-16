import networkx as nx
import numpy as np
from scipy import special
from abeba_methods import compute_activation, compute_post

class RecommendedFriendNotFound(Exception):
    """Raised when the People Recommender fails to recommend a new friend to a given node"""
    pass

class SimulateEpochPeopleRecommenderError(Exception):
    """Raised when an error occurred in the simulate_epoch_people_recommender method"""
    pass

class SubstrategyNotRecognized(Exception):
    """Raised when the param 'substrategy' has a not regnized value and chosen strategy isn't random"""
    pass

class StrategyNotRecognized(Exception):
    """Raised when the param 'strategy' has a not regnized value"""
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
""" def x_hop_neighbors(G, nodes, x_hop, previous_nodes = None):
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
        return x_hop_neighbors(G, all_new_friends_list_unique, x_hop-1, previous_nodes) """

def x_hop_neighbors(G, nodes, x_hop):
    all_friends_list = []
    for node in nodes:
        friends = list(nx.neighbors(G, node))
        all_friends_list += friends
        all_friends_list_unique = list(dict.fromkeys(all_friends_list))
    if x_hop == 1:
        return all_friends_list_unique
    else:
        return x_hop_neighbors(G, all_friends_list_unique, x_hop-1)



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
def people_recommender(G, nodes, strategy="random", substrategy=None):
    if strategy == "opinion_estimation_based" or strategy == "topology_based": 
        if substrategy != "counteract_homophily" or substrategy != "favour_homophily":
            raise SubstrategyNotRecognized
    elif strategy != "random": 
       raise StrategyNotRecognized 

    all_nodes = list(G.nodes)
    for node_id in nodes:
        recommended_friend = None
        neigs = list(nx.neighbors(G, node_id))
        not_friends = [x for x in all_nodes if x not in neigs]
        not_friends.remove(node_id)
        if strategy == "random":
            # recommending a random node not already friend as a new friend. 
            recommended_friend = np.random.choice(not_friends, size=1, replace=False)
        elif strategy == 'opinion_estimation_based':
            nodes_with_estimated_opinion = list(nx.get_node_attributes(G, name='estimated_opinion').keys())
            posteri_errors_dict = nx.get_node_attributes(G, name='posteri_error')
            distances_dict = {}
            for key in nodes_with_estimated_opinion:
                if key in not_friends:
                    abs_distance = abs(nodes_with_estimated_opinion[node_id] - nodes_with_estimated_opinion[key])
                    if bool(posteri_errors_dict):
                        posteri_error_subject = posteri_errors_dict[node_id]
                        posteri_errors_suggested_node = posteri_errors_dict[key]
                        # it prevents division by zero and sets a minimum achievable value
                        if posteri_error_subject >= 0 and posteri_error_subject <= 0.000000001:
                            posteri_error_subject = 0.000000001
                        if posteri_errors_suggested_node >= 0 and posteri_errors_suggested_node <= 0.000000001:
                            posteri_errors_suggested_node = 0.000000001
                        distances_dict[key] = abs_distance / (posteri_error_subject + posteri_errors_suggested_node)
                    else:
                        distances_dict[key] = abs_distance
            if substrategy == "counteract_homophily":
                distances_distribution = special.softmax(distances_dict.values())
            elif substrategy == "favour_homophily":
                # it penalizes the furthest nodes. Note that this is exactly like having 2 - abs_distance in the previous calculation
                distances_distribution = special.softmax([-1*x for x in distances_dict.values()])
            else:
                raise SubstrategyNotRecognized
            recommended_friend = np.random.choice(distances_dict.keys(), size=1, replace=False, p=distances_distribution)
        elif strategy == 'topology_based':
            if substrategy == "favour_homophily":
                overlapping_dict = {}
                # BFS 
                visited, queue = [[] for _ in range(4)], []
                visited[0].append(node_id)
                queue.append((node_id, 0))
                while queue:
                    # next pair (node, distance)
                    next = queue.pop(0)
                    # reached max distance
                    if next[1] == 3:
                        break
                    next_neigs = list(nx.neighbors(G, next[0]))
                    for neig in next_neigs:
                        if neig not in visited[next[1] + 1]:
                            visited[next[1] + 1].append(neig)
                            queue.append((neig, next[1] + 1))
                # BFS ending
                hop3 = [x for x in visited[3] if x not in neigs and x != node_id]
                for not_friend in hop3:
                    not_friend_neigs = list(nx.neighbors(G, not_friend))
                    # the number of mutual friendly nodes is given by the length of the intersection between the two friend lists
                    # note that neither of the two lists can have duplicates
                    number_overlapping_friends = len(set(not_friend_neigs) & set(neigs))
                    overlapping_dict[not_friend] = number_overlapping_friends

                overlapping_distribution = special.softmax(overlapping_dict.values())
                recommended_friend = np.random.choice(overlapping_dict.keys(), size=1, replace=False, p=overlapping_distribution)
            elif substrategy == "counteract_homophily":
                # BFS
                visited, queue = [], []
                visited.append(node_id)
                queue.append(node_id)
                dist = {}
                dist[node_id] = 0
                while queue:
                    next = queue.pop(0)
                    next_neigs = list(nx.neighbors(G, next))
                    for neig in next_neigs:
                        if neig not in visited:
                            visited.append(neig)
                            queue.append(neig)
                            dist[neig] = dist[next] + 1

                dist_not_friends = { key:value for (key,value) in dist.items() if key in not_friends}
                distances_distribution = special.softmax(dist_not_friends.values())
                recommended_friend = np.random.choice(dist_not_friends.keys(), size=1, replace=False, p=distances_distribution)
            else:
                raise SubstrategyNotRecognized
            
        if recommended_friend is None:
            raise RecommendedFriendNotFound
        else:
            nx.set_node_attributes(G, {node_id: recommended_friend[0]}, 'person_recommended')
            # note that recommended_friend is a numpy array with 1 element
            G.add_edge(node_id, recommended_friend[0])
            # it takes the neighbors again (there will also be the one just added)
            neigs = list(nx.neighbors(G, node_id))
            # deleting a random edge to prevent fully connected graphs.
            # it only keeps neighbors who have at least two friends (node_id and another)
            # otherwise, the unpopular nodes would be slowly isolated from the others.
            neigs_popular = [neig for neig in neigs if len(list(nx.neighbors(G, neig))) > 1]
            if neigs_popular:
                discarded_friend = np.random.choice(neigs_popular, size=1, replace=False)
                # note that discarded_friend is a numpy array with 1 element
                G.remove_edge(node_id, discarded_friend[0])
            else:
                print("WARNING: node " + node_id + " has only neighbors who have only him as a friend, so no edges have been cut")
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
  except (RecommendedFriendNotFound, StrategyNotRecognized, SubstrategyNotRecognized):
    print("the People Recommender failed to recommend a new friend to a given node")
    raise SimulateEpochPeopleRecommenderError

  return G
