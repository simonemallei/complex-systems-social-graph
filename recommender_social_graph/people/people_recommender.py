import networkx as nx
import numpy as np
import copy
import random
from scipy import special
from abeba_methods import compute_activation, compute_post
from estimation import upd_estim

class RecommendedFriendNotFound(Exception):
    """Raised when the People Recommender fails to recommend a new friend to a given node"""
    pass

class PeopleRecommenderError(Exception):
    """Raised when an error occurred in the people_recommender method"""
    pass

class SimulateEpochPeopleRecommenderError(Exception):
    """Raised when an error occurred in the simulate_epoch_people_recommender method"""
    pass

class SubstrategyNotRecognized(Exception):
    """Raised when the param 'substrategy' has a not regnized value and chosen strategy isn't random"""
    pass

class SubstrategyError(Exception):
    """Raised when the current substrategy cannot recommend any people"""
    pass

class StrategyNotRecognized(Exception):
    """Raised when the param 'strategy' has a not regnized value"""
    pass

class StrategyOpinionEstimationBasedError(Exception):
    """Raised when an error occurred in the strategy_opinion_estimation_based method"""
    pass

class StrategyTopologyBasedError(Exception):
    """Raised when an error occurred in the strategy_topology_based method"""
    pass

'''
strategy_opinion_estimation_based applies a strategy based on the estimation of the nodes' 
opinions: the distances between the estimation of the opinion of the current node and that 
of the other nodes are calculated. The error of the estimations is also used if present 
(i.e. if the kalman filter strategy was used to calculate the estimate). Some nodes will 
have no opinion, and therefore cannot be recommended. The closest or furthest opinion 
is favored depending on whether the chosen sub-strategy is favor homophily or counteract 
homophily.
Note that it is impossible that there are no estimations of the nodes' opinions, 
unless a zero value of nodes that can publish content has been set in the model. 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    substrategy : {string}
        Possible values are: counteract_homophily, favour_homophily
    neigs : {list of ints}
        List of friend node ids of the current node
    node_id : {int}
        current node id

Returns
-------
    : {dict}         
        It contains a probability distribution on candidate nodes to be recommended.
'''

def strategy_opinion_estimation_based(G, substrategy, neigs, node_id):
    nodes_with_estimated_opinion = nx.get_node_attributes(G, name='estimated_opinion')
    posteri_errors_dict = nx.get_node_attributes(G, name='posteri_error')
    distances_dict = {}
    for key in nodes_with_estimated_opinion.keys():
        if key not in neigs and key != node_id:
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
        if distances_dict:
            distances_distribution = special.softmax(list(distances_dict.values()))
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('An error occurred in substrategy ' + substrategy)
                raise StrategyOpinionEstimationBasedError
    elif substrategy == "favour_homophily":
        if distances_dict:
            # it penalizes the furthest nodes. Note that this is exactly like having 2 - abs_distance in the previous calculation
            distances_distribution = special.softmax([-1*x for x in distances_dict.values()])
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('An error occurred in substrategy ' + substrategy)
                raise StrategyOpinionEstimationBasedError
    else:
        try:
            raise SubstrategyNotRecognized
        except SubstrategyNotRecognized:
            raise StrategyOpinionEstimationBasedError

    #recommended_friend = np.random.choice(list(distances_dict.keys()), size=1, replace=False, p=distances_distribution)
    return dict(zip(distances_dict.keys(), distances_distribution))


'''
strategy_topology_based applies the topology based strategy based on the chosen sub-strategy.
If the sub-strategy is favour_homophily, then nodes at distance 3 from the current node (3-hops) 
are taken by a modifiend BFS algorithm and mutual friends between these nodes and the current node 
are counted. Based on these values, a probability distribution is assigned that favors nodes with 
more friends in common with the current node.
If the sub-strategy is counteract homophily, then the nodes furthest away from the current node are 
taken by a modifiend BFS algorithm and the distance is counted based on the number of nodes crossed.
Based on these values, a probability distribution is assigned that favors the furthest nodes away 
from the current node.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    substrategy : {string}
        Possible values are: counteract_homophily, favour_homophily
    neigs : {list of ints}
        List of friend node ids of the current node
    node_id : {int}
        current node id

Returns
-------
    res_dict : {dict}         
            It contains a probability distribution on candidate nodes to be recommended.
'''

def strategy_topology_based(G, substrategy, neigs, node_id):
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

        if overlapping_dict:
            overlapping_distribution = special.softmax(list(overlapping_dict.values()))
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('An error occurred in substrategy ' + substrategy)
                raise StrategyTopologyBasedError
        res_dict = dict(zip(overlapping_dict.keys(), overlapping_distribution))

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

        dist_not_friends = { key:value for (key,value) in dist.items() if key not in neigs and key != node_id}

        if dist_not_friends:
            distances_distribution = special.softmax(list(dist_not_friends.values()))
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('An error occurred in substrategy ' + substrategy)
                raise StrategyTopologyBasedError
        res_dict = dict(zip(dist_not_friends.keys(), distances_distribution))

    else:
        try:
            raise SubstrategyNotRecognized
        except SubstrategyNotRecognized:
            raise StrategyTopologyBasedError

    return res_dict


'''
breadth_first_search performs a simple BFS algorithm to find all reachable 
nodes from the {source_node} node.
This method is used to discover all nodes of each isolated component of the graph

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    source_node : {int}
        The source node ID. The exploration of the graph will start from this node

Returns
-------
    visited : {list of ints}         
            The list containing all found nodes
'''

def breadth_first_search(G, source_node):
    # BFS
    visited, queue = [], []
    visited.append(source_node)
    queue.append(source_node)
    while queue:
        next = queue.pop(0)
        next_neigs = list(nx.neighbors(G, next))
        for neig in next_neigs:
            if neig not in visited:
                visited.append(neig)
                queue.append(neig)
    return visited

'''
people_recommender performs a people recommender system.
It performs the recommendation routine on the nodes contained 
in {nodes}.
The recommender routine depends on which strategy and substrategy are chosen 
(by default, the method use the "random" one). 
At the beginning of the method code, the recommendation data of the previous epoch 
concerning all those nodes that do not publish content in this epoch are deleted. 
Instead, the data of those nodes that will publish content in this epoch are overwritten 
as soon as the routine chooses the new node.
Then, if the topology based strategy has been chosen, a fake graph is created starting 
from the real one. In this fake graph the isolated components of the real graph are 
connected together and then other random edges are inserted up to a total of 5% of 
inserted false edges. This graph will be kept for the entire epoch and used by the 
recommender to explore new nodes to recommend. Note that it remains the same for all 
nodes acted upon by the recommender in the current epoch.

Note that, at the end of the method, a node is removed from friends to avoid getting a 
fully connected graph. As the code is currently implemented, this node cannot be the 
one just added to friends, nor can it be a friend node that has no other friends. This 
last constraint serves to avoid penalizing nodes with few friends, which, otherwise, 
risk being isolated.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    nodes : {list of object}
        The list containing nodes' IDs (dictionary keys) on which to run the people recommender
    strategy : {String} default: "random"
        The string that defines the strategy used by the recommender system.
        There are two possible strategies that can be combined with two possible sub-strategies,
        in addition to the random strategy:
        Strategies: opinion_estimation_based, topology_based
    Substrategies: {String} default: None
        Possible values are: counteract_homophily, favour_homophily

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def people_recommender(G, nodes, strategy="random", substrategy=None):
    # Deletes the data of the previous epoch.
    # For performance reasons, the attribute of the nodes on which to run the recommendation system is not deleted
    # because it will be overwritten
    person_recommended_dict = nx.get_node_attributes(G, name='person_recommended')
    for key in person_recommended_dict.keys():
        if key not in nodes:
            del G.nodes[key]['person_recommended']

    all_nodes = list(G.nodes)

    # Preparing G_fake for topology based strategy
    if strategy == 'topology_based': 
        # before launching the BFS for the selected sub-strategy, we look for all the disconnected components 
        # of the graph to connect them. Note that the resulting graph will only be used on the all posting nodes but only in current epoch.
        number_components = nx.number_connected_components(G)
        G_fake = None
        # the components are reconnected only if there are at least 2
        if number_components > 1:
            G_fake = copy.deepcopy(G)
            old_nodes_found = []
            nodes_not_found = all_nodes
            # No need to explore the last component. The nodes that have not yet been found are the only ones present in it
            for _ in range(number_components-1):
                chosen_node = np.random.choice(nodes_not_found, size=1, replace=False) 
                # performs an exploration with BFS Algorithm
                nodes_found = breadth_first_search(G, chosen_node[0])
                if old_nodes_found:
                    node_source = np.random.choice(old_nodes_found, size=1, replace=False)
                    node_target = np.random.choice(nodes_found, size=1, replace=False)
                    G_fake.add_edge(node_source[0], node_target[0])
                old_nodes_found += nodes_found
                nodes_not_found = list(set(nodes_not_found) - set(old_nodes_found))
            
            # Now there is only one component left to connect, which nodes know about without having to explore it
            node_source = np.random.choice(old_nodes_found, size=1, replace=False)
            node_target = np.random.choice(nodes_not_found, size=1, replace=False)
            G_fake.add_edge(node_source[0], node_target[0])
            
        # The total number of arches added, in the end, will be equal to 5% of the total number of arches
        number_edges_to_insert = (5 * G.number_of_edges() // 100) - (number_components - 1)
        for _ in range(number_edges_to_insert):
            chosen_nonedge  = random.choice(list(nx.non_edges(G_fake)))
            G_fake.add_edge(chosen_nonedge[0], chosen_nonedge[1])

    for node_id in nodes:
        recommended_friend = None
        neigs = list(nx.neighbors(G, node_id))
        if strategy == "random":
            # recommending a random node not already friend as a new friend. 
            recommended_friend = np.random.choice([x for x in all_nodes if x not in neigs and x != node_id], size=1, replace=False)
        elif strategy == 'opinion_estimation_based':
            try:
                res_dict = strategy_opinion_estimation_based(G, substrategy, neigs, node_id)
            except StrategyOpinionEstimationBasedError:
                raise PeopleRecommenderError
        elif strategy == 'topology_based':
            try:
                if G_fake is not None:
                    res_dict = strategy_topology_based(G_fake, substrategy, neigs, node_id)
                else:
                    res_dict = strategy_topology_based(G, substrategy, neigs, node_id)
            except StrategyTopologyBasedError:
                raise PeopleRecommenderError
        else:
            try:
                raise StrategyNotRecognized
            except StrategyNotRecognized:
                raise PeopleRecommenderError

        if recommended_friend is None:
            recommended_friend = np.random.choice(list(res_dict.keys()), size=1, replace=False, p=list(res_dict.values()))

        nx.set_node_attributes(G, {node_id: recommended_friend[0]}, 'person_recommended')
        # note that recommended_friend is a numpy array with 1 element
        G.add_edge(node_id, recommended_friend[0])
        # deleting a random edge to prevent fully connected graphs.
        # it only keeps neighbors who have at least two friends (node_id and another)
        # otherwise, the unpopular nodes would be slowly isolated from the others.
        neigs_popular = [neig for neig in neigs if len(list(nx.neighbors(G, neig))) > 1]
        if neigs_popular:
            discarded_friend = np.random.choice(neigs_popular, size=1, replace=False)
            # note that discarded_friend is a numpy array with 1 element
            G.remove_edge(node_id, discarded_friend[0])
        else:
            print("WARNING: node " + str(node_id) + " hasn't friends (except the one just added) or has only neighbors who have only him as a friend, so no edges have been cut")
    return G

'''
simulate_epoch_people_recommender simulates an epoch. It randomly activates a 
percentage ({percent_updating_nodes}) of graph's nodes and so each node will 
update its opinion base on its feed.
Afterwards, a percentage of the activated nodes (always sampled randomly) will 
also be posting nodes, updating their neighbours' feed with the content. The 
opinion shared by the posting nodes has a noise related to the parameter {epsilon}.
Then the people recommender is run on the posting_nodes.

IMPORTANT: this method is only used to test the People Recommender by isolating it 
from the Content Recommender. This method is not what is actually used for simulations.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    rate_updating_nodes : {float}
        The percentage of the nodes that will be activated. Interval [0,1]
    epsilon : {float} default: 0.0
        The Gaussian noise's standard deviation in the posting phase.
    estim_strategy: {String} default: "base"
        The strategy that is used to estimate the opinion of the nodes that posting a content
    estim_strategy_param : {dictionary}
        The dictionary containing the parameters value used by the recommender, based
        on {strategy} value.
            - key: "alpha",
              value: alpha coefficient used in "base" strategy. Default: 0.9
            - key: "variance",
              value: process variance of "kalman" strategy. Default: 1e-5
            - key: "variance_measure",
              value: measure variance of "kalman" strategy. Default: 0.1 ** 2 (0.1 ^ 2)
    strategy_people_recommender : {String} default: "random"
        The string that defines the strategy used by the recommender system.
        There are two possible strategies that can be combined with two possible sub-strategies,
        in addition to the random strategy:
        Strategies: opinion_estimation_based, topology_based
    Substrategies: {String} default: None
        Possible values are: counteract_homophily, favour_homophily

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def simulate_epoch_people_recommender(
    G, 
    rate_updating_nodes, 
    epsilon = 0.0, 
    estim_strategy = "base", 
    estim_strategy_param = {},
    strategy_people_recommender = "random",
    substrategy_people_recommender = None
    ):

    # Sampling randomly the activating nodes
    updating_nodes = int(rate_updating_nodes * len(G.nodes()))
    act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)

    # Executing activation phase: activated nodes will consume their feed
    G = compute_activation(G, act_nodes)

    # Executing posting phase: activated nodes will post in their neighbours' feed
    G, posting_nodes_list = compute_post(G, act_nodes, epsilon)
    # Estimating opinion by the recommender
    G = upd_estim(G, posting_nodes_list, strategy = estim_strategy, strat_param = estim_strategy_param)
    try:
      G = people_recommender(G, posting_nodes_list, strategy_people_recommender, substrategy_people_recommender)
    except PeopleRecommenderError:
      print("the People Recommender failed to recommend a new friend to a given node")
      raise SimulateEpochPeopleRecommenderError
    return G