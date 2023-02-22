import networkx as nx
import numpy as np
import copy
import random
import time
from collections import deque
from scipy import special
from abeba_methods import compute_activation, compute_post
from estimation import upd_estim
from util import softmax_with_temperature

SOFTMAX_TEMPERATURE = 0.03

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

class StrategyOpinionEstimationTopologyMixedError(Exception):
    """Raised when an error occurred in the strategy_opinion_estimation_topology_mixed method"""
    pass

class StratParamIsNotEmpty(Exception):
    """Raised when strat_param is not empty and the selected strategy is not opinion_estimation_topology_mixed"""
    pass

'''
strategy_opinion_estimation_based applies a strategy based on the estimation of the nodes' 
opinions: the distances between the estimation of the opinion of the current node and that 
of the other nodes are calculated. The error of the estimations is also used
(note that only the opinion estimation strategy using the kalman filter will change the value).  
The closest or furthest opinion is favored depending on whether the chosen sub-strategy is 
favor homophily or counteract homophily.
Note that each node will always have an estimation of its opinion, because for each node a starting 
value is fixed which will then be updated as the epochs go by.

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
            posteri_error_subject = posteri_errors_dict[node_id]
            posteri_errors_suggested_node = posteri_errors_dict[key]
            # it prevents division by zero and sets a minimum achievable value
            if posteri_error_subject >= 0 and posteri_error_subject <= 0.000000001:
                posteri_error_subject = 0.000000001
            if posteri_errors_suggested_node >= 0 and posteri_errors_suggested_node <= 0.000000001:
                posteri_errors_suggested_node = 0.000000001
            distances_dict[key] = abs_distance / (posteri_error_subject + posteri_errors_suggested_node)
    if substrategy == "counteract_homophily":
        if distances_dict:
            distances_distribution = softmax_with_temperature(logits=list(distances_dict.values()), temperature=SOFTMAX_TEMPERATURE)
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('ERROR! An error occurred in substrategy: ' + substrategy+ '\n')
                raise StrategyOpinionEstimationBasedError
    elif substrategy == "favour_homophily":
        if distances_dict:
            # it penalizes the furthest nodes. Note that this is exactly like having 2 - abs_distance in the previous calculation
            distances_distribution = softmax_with_temperature(logits=[-1*x for x in distances_dict.values()], temperature=SOFTMAX_TEMPERATURE)
        else:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('ERROR! An error occurred in substrategy: ' + substrategy + '\n')
                raise StrategyOpinionEstimationBasedError
    else:
        try:
            raise SubstrategyNotRecognized
        except SubstrategyNotRecognized:
            print('ERROR! Substrategy not recognized\n')
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
    res_dict_can_be_empty : {boolean}
        Default False. If False, the code will throw an exception if no nodes were found that can be 
        recommended to node {node_id}. If true, then the exception will not be thrown. The latter case is 
        important if the connected_components parameter of the people recommender is set to 0 to prevent 
        exceptions from being triggered when the {node_id} node is isolated (see the strat_param parameter).
    all_neighbors: {dict}
        the dictionary containing the neighbors set given a node

Returns
-------
    res_dict : {dict}         
            It contains a probability distribution on candidate nodes to be recommended.
'''

def strategy_topology_based(G, substrategy, neigs, node_id, res_dict_can_be_empty = False, all_neighbors = {}):
    if substrategy == "favour_homophily":
        overlapping_dict = {}
        # BFS 
        visited, queue = [set() for _ in range(4)], deque()
        visited[0].add(node_id)
        queue.append((node_id, 0))
        while queue:
            # next pair (node, distance)
            next = queue.popleft()
            # reached max distance
            if next[1] == 3:
                break
            to_add = all_neighbors[next[0]] - visited[next[1] + 1]
            for neig in to_add:
                visited[next[1] + 1].add(neig)
                queue.append((neig, next[1] + 1))
                
        # BFS ending
        hop3 = tuple(x for x in visited[3] if x not in neigs and x != node_id)
        for not_friend in hop3:
            not_friend_neigs = all_neighbors[not_friend]
            # the number of mutual friendly nodes is given by the length of the intersection between the two friend lists
            # note that neither of the two lists can have duplicates
            number_overlapping_friends = len(not_friend_neigs & neigs)
            overlapping_dict[not_friend] = number_overlapping_friends

        overlapping_distribution = []
        if overlapping_dict:
            overlapping_distribution = softmax_with_temperature(logits=list(overlapping_dict.values()), temperature=SOFTMAX_TEMPERATURE)
        elif not res_dict_can_be_empty:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('ERROR! An error occurred in substrategy: ' + substrategy + '\n')
                raise StrategyTopologyBasedError
        res_dict = dict(zip(overlapping_dict.keys(), overlapping_distribution))

    elif substrategy == "counteract_homophily":
        # BFS
        visited, queue = set(), deque()
        visited.add(node_id)
        queue.append(node_id)
        dist = {}
        dist[node_id] = 0
        while queue:
            next = queue.popleft()
            to_add = all_neighbors[next] - visited
            for neig in to_add:
                visited.add(neig)
                queue.append(neig)
                dist[neig] = dist[next] + 1

        dist_not_friends = { key:value for (key,value) in dist.items() if key not in neigs and key != node_id}

        distances_distribution = []
        if dist_not_friends:
            distances_distribution = softmax_with_temperature(logits=list(dist_not_friends.values()), temperature=SOFTMAX_TEMPERATURE)
        elif not res_dict_can_be_empty:
            try:
                raise SubstrategyError
            except SubstrategyError:
                print('ERROR! An error occurred in substrategy: ' + substrategy + '\n')
                raise StrategyTopologyBasedError
        res_dict = dict(zip(dist_not_friends.keys(), distances_distribution))

    else:
        try:
            raise SubstrategyNotRecognized
        except SubstrategyNotRecognized:
            print('ERROR! Substrategy not recognized\n')
            raise StrategyTopologyBasedError

    return res_dict

'''
strategy_opinion_estimation_topology_mixed applies the opinion_estimation_topology_mixed 
strategy: it launches the two strategies opinion_estimation_based and topology_based, 
and puts the results together by taking the position of the recommendable nodes, ordered 
by probability of extraction. If the same node is present in the recommendable list of both 
strategies, its positions are averaged. Otherwise only the single existing position is taken. 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    G_top_strat : {networkx.Graph}
        The graph used in topology based strategy. It can contain all the joined components of the graph G, with a possible 
        addition of other edges up to 5% of the total, or the {G} graph. It contains G graph if number of G components is equal 
        to 1 and if there aren't new edges to add.
    substrategy : {string}
        Possible values are: counteract_homophily, favour_homophily
    neigs : {list of ints}
        List of friend node ids of the current node
    node_id : {int}
        current node id
    all_neighbors: {dict}
        the dictionary containing the neighbors set given a node
    connected_components: {boolean}
        Default True. If the value is True, then the opinion_estimation_topology_mixed strategy will connect the components 
        of the graph before choosing who to recommend (using both main strategies), as is the case for the topology_based strategy.
        If the value is False, the opinion_estimation_topology_mixed strategy will always use both main strategies, but the 
        contribution of the topology_based strategy will be limited only to the nodes present in the considered component.

Returns
-------
    res_dict : {dict}         
            It contains a probability distribution on candidate nodes to be recommended.
'''

def strategy_opinion_estimation_topology_mixed(G, G_top_strat, substrategy, neigs, node_id, all_neighbors, connected_components=True):
    if (substrategy == "favour_homophily") or (substrategy == "counteract_homophily"):
        try:
            ordered_op_estim_based_dict = {k: v for k, v in sorted(strategy_opinion_estimation_based(G, substrategy, neigs, node_id).items(), key=lambda item: item[1], reverse=True)}
            ordered_top_based_dict = {k: v for k, v in sorted(strategy_topology_based(G_top_strat, substrategy, neigs, node_id, True if connected_components is False else False, all_neighbors=all_neighbors).items(), key=lambda item: item[1], reverse=True)}
        except (StrategyOpinionEstimationBasedError, StrategyTopologyBasedError) as error:
            print(error)
            raise StrategyOpinionEstimationTopologyMixedError
        else:
            op_estim_candidates = ordered_op_estim_based_dict.keys()
            top_candidates = ordered_top_based_dict.keys()
            all_candidates = list(set(op_estim_candidates) | set(top_candidates))
            positions_mixed_dict = {}
            for candidate in all_candidates:
                position_op_estim = None
                position_top = None
                if (candidate in op_estim_candidates) and (candidate in top_candidates):
                    position_op_estim = list(op_estim_candidates).index(candidate) + 1
                    position_top = list(top_candidates).index(candidate) + 1
                    position = (position_op_estim + position_top) / 2
                elif (candidate in op_estim_candidates) and (candidate not in top_candidates):
                    position = list(op_estim_candidates).index(candidate) + 1
                else:
                    position = list(top_candidates).index(candidate) + 1

                positions_mixed_dict[candidate] = position
            
            if positions_mixed_dict:
                positions_distribution = softmax_with_temperature(logits= list(positions_mixed_dict.values()), temperature=SOFTMAX_TEMPERATURE)
            else:
                raise StrategyOpinionEstimationTopologyMixedError
            res_dict = dict(zip(positions_mixed_dict.keys(), positions_distribution))
    else:
        try:
            raise SubstrategyNotRecognized
        except SubstrategyNotRecognized:
            print('ERROR! Substrategy not recognized\n')
            raise StrategyOpinionEstimationTopologyMixedError

    return res_dict

'''
people_recommender performs a people recommender system.
It performs the recommendation routine on the nodes contained 
in {nodes}.

At the beginning of the method code, a check is made on the {strat_param} parameter: 
in fact, it is only used together with the opinion_estimation_topology_mixed strategy.
That parameter allows to use or not the hypothesis according to which there is noise in 
the graph which leads to a lack of some connections between nodes with similar opinions.
Then, the recommendation data of the previous epoch are deleted. 
Then, if the topology based strategy has been chosen, it is assumed that the {strat_param}
parameter is set to True, so a fake graph is created starting from the real one 
(otherwise there is a risk of failing to recommend any node to the current node).
In this fake graph the isolated components of the real graph are 
connected together and then other random edges are inserted up to a total of 5% of 
inserted false edges. This graph will be kept for the entire epoch and used by the 
recommender to explore new nodes to recommend. Note that it remains the same for all 
nodes acted upon by the recommender in the current epoch.
Finally, the correct method is launched based on the selected strategy. (by default, the 
method use the "random" one). Among the various parameters, the selected sub-strategy will 
also be passed to it.
Finally, the data on the new nodes to recommend that has been collected is saved in the graph 
and the new arcs are created. Note that, at the end of the method, a node is removed from 
friends to avoid getting a fully connected graph. As the code is currently implemented, 
this node cannot be the one just added to friends, nor can it be a friend node that has no 
other friends. This last constraint serves to avoid penalizing nodes with few friends, which, 
otherwise, risk being isolated.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    nodes : {list of object}
        The list containing nodes' IDs (dictionary keys) on which to run the people recommender
    strategy : {String} default: "random"
        The string that defines the strategy used by the recommender system.
        There are several possible strategies that can be combined with two possible sub-strategies:
        Strategies: no_recommender, random, opinion_estimation_based, topology_based, opinion_estimation_topology_mixed
    Substrategies: {String} default: None
        Possible values are: counteract_homophily, favour_homophily. They can be used with the following strategies:
        opinion_estimation_based, topology_based, opinion_estimation_topology_mixed. Parameter ignored by other strategies
    strat_param: {dictionary} default: {"connected_components": True}
        dictionary that containing the parameters value used by the recommender. In the current version, the only strategy using this dictionary is opinion_estimation_topology_mixed. 
        Elements:
        Connected_components: False or True default: True
        If the value is True, then the opinion_estimation_topology_mixed strategy will connect the components of the graph before choosing who to recommend (using both main strategies), as is the case for the topology_based strategy.
        If the value is False, the opinion_estimation_topology_mixed strategy will always use both main strategies, but the contribution of the topology_based strategy will be limited only to the nodes present in the considered component.

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def people_recommender(G, nodes, strategy="random", substrategy=None, strat_param={}):
    if strategy != "opinion_estimation_topology_mixed" and strat_param:
        try:
            raise StratParamIsNotEmpty
        except StratParamIsNotEmpty:
            raise PeopleRecommenderError

    # Deletes the data of the previous epoch.
    old_person_recommended_dict = nx.get_node_attributes(G, name='person_recommended')
    for key in old_person_recommended_dict.keys():
        del G.nodes[key]['person_recommended']

    old_discarded_friend_dict = nx.get_node_attributes(G, name='discarded_friend')
    for key in old_discarded_friend_dict.keys():
        del G.nodes[key]['discarded_friend']

    # Initialize data
    person_recommended_dict = {}
    all_nodes = list(G.nodes)

    # Preparing G_fake for topology based strategy
    G_fake = None
    if strategy == 'topology_based' or (strategy == 'opinion_estimation_topology_mixed' and strat_param.get('connected_components', True)): 
        # before launching the BFS for the selected sub-strategy, we look for all the disconnected components 
        # of the graph to connect them. Note that the resulting graph will only be used on the all posting nodes but only in current epoch.
        number_components = nx.number_connected_components(G)
        # the components are reconnected only if there are at least 2
        if number_components > 1:
            components = nx.connected_components(G)
            G_fake = nx.Graph()
            G_fake.add_nodes_from(next(components))
            G_fake.add_edges_from(G.edges(G_fake.nodes))
            # note that for starts from second component because of next(components)
            for nodes_component in components:
                node_source = np.random.choice(list(G_fake.nodes), size=1, replace=False)
                node_target = np.random.choice(list(nodes_component), size=1, replace=False)
                G_fake.add_nodes_from(nodes_component)
                G_fake.add_edges_from(G.edges(nodes_component))
                G_fake.add_edge(node_source[0], node_target[0])

        # The total number of arches added, in the end, will be equal to 5% of the total number of arches
        number_edges_to_insert = (5 * G.number_of_edges() // 100) - (number_components - 1)

        if number_edges_to_insert > 0:
            non_edges_list = list(nx.non_edges(G))
            if G_fake is None:
                G_fake = copy.deepcopy(G)
            if number_edges_to_insert < len(non_edges_list):
                chosen_nonedges_keys = set(np.random.choice(len(non_edges_list), size=number_edges_to_insert, replace=False))
                chosen_nonedge = [non_edges_list[index] for index in chosen_nonedges_keys]
                G_fake.add_edges_from(chosen_nonedge)
            else:
                G_fake.add_edges_from(non_edges_list)

    # Use graph for topology based strategy based on whether G_fake is defined or not
    G_top_strat = G if G_fake is None else G_fake
    all_neighbors = {
        node: set(nx.neighbors(G_top_strat, node))
        for node in G_top_strat.nodes()
    }

    for node_id in nodes:
        recommended_friend = None
        neigs = all_neighbors[node_id]
        if strategy == "random":
            res_dict = None
        elif strategy == 'opinion_estimation_based':
            try:
                res_dict = strategy_opinion_estimation_based(G, substrategy, neigs, node_id)
            except StrategyOpinionEstimationBasedError:
                print('ERROR! An error occurred in strategy_opinion_estimation_based method\n')
                raise PeopleRecommenderError
        elif strategy == 'topology_based':
            try:
                res_dict = strategy_topology_based(G_top_strat, substrategy, neigs, node_id, all_neighbors=all_neighbors)
            except StrategyTopologyBasedError:
                print('ERROR! An error occurred in strategy_topology_based method\n')
                raise PeopleRecommenderError
        elif strategy == 'opinion_estimation_topology_mixed':
            try:
                res_dict = strategy_opinion_estimation_topology_mixed(G, G_top_strat, substrategy, neigs, node_id, all_neighbors=all_neighbors, connected_components=strat_param.get('connected_components', True))
            except StrategyOpinionEstimationTopologyMixedError:
                print('ERROR! An error occurred in strategy_opinion_estimation_topology_mixed method\n')
                raise PeopleRecommenderError
        elif strategy == "no_recommender":
            # continue to the next for value
            continue
        else:
            try:
                raise StrategyNotRecognized
            except StrategyNotRecognized:
                print('ERROR! Strategy not recognized\n')
                raise PeopleRecommenderError

        # if res_dict is None, strategy is random, therefore the choice takes place between nodes that are not friends of the node_id node and that are not the node itself, with uniform distribution
        recommended_friend = np.random.choice([x for x in all_nodes if x not in neigs and x != node_id] if res_dict is None else list(res_dict.keys()), size=1, replace=False, p= None if res_dict is None else list(res_dict.values()))
        # note that recommended_friend is a not json serializable numpy array with 1 element, while python int type is json serializable
        person_recommended_dict[node_id] = int(recommended_friend[0])

    nx.set_node_attributes(G, person_recommended_dict, 'person_recommended')
    discarded_friend_dict = {}
    for key in person_recommended_dict.keys():
        G.add_edge(key, person_recommended_dict[key])
        # deleting a random edge to prevent fully connected graphs.
        # it only keeps neighbors who have at least two friends (node_id and another)
        # and that they are not the newly added node,
        # otherwise the unpopular nodes would be slowly isolated from the others.
        neigs = [neig for neig in list(nx.neighbors(G, key)) if neig != person_recommended_dict[key]]
        neigs_popular = [neig for neig in neigs if len(list(nx.neighbors(G, neig))) > 1]
        if neigs_popular:
            discarded_friend = np.random.choice(neigs_popular, size=1, replace=False)
            # note that discarded_friend is a not json serializable numpy array with 1 element, while python int type is json serializable
            discarded_friend_dict[key] = int(discarded_friend[0])
            G.remove_edge(key, discarded_friend[0])
        #else:
            #print("WARNING: node " + str(key) + " hasn't friends (except the one just added) or has only neighbors who have only him as a friend, so no edges have been cut\n")

    nx.set_node_attributes(G, discarded_friend_dict, 'discarded_friend')
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
        There are several possible strategies that can be combined with two possible sub-strategies:
        Strategies: no_recommend, random, opinion_estimation_based, topology_based, opinion_estimation_topology_mixed
    substrategy_people_recommender: {String} default: None
        Possible values are: counteract_homophily, favour_homophily. They can be used with the following strategies:
        opinion_estimation_based, topology_based, opinion_estimation_topology_mixed. Parameter ignored by other strategies
    people_recomm_strat_param: {dictionary} default: {"connected_components": True}
        dictionary that containing the parameters value used by the recommender. In the current version, the only strategy using this dictionary is opinion_estimation_topology_mixed. 
        Elements:
        Connected_components: False or True default: True
        If the value is True, then the opinion_estimation_topology_mixed strategy will connect the components of the graph before choosing who to recommend (using both main strategies), as is the case for the topology_based strategy.
        If the value is False, the opinion_estimation_topology_mixed strategy will always use both main strategies, but the contribution of the topology_based strategy will be limited only to the nodes present in the considered component.

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
    substrategy_people_recommender = None,
    people_recomm_strat_param = {}
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
      G = people_recommender(G, posting_nodes_list, strategy_people_recommender, substrategy_people_recommender, strat_param=people_recomm_strat_param)
    except PeopleRecommenderError:
      print("the People Recommender failed to recommend a new friend to a given node")
      raise SimulateEpochPeopleRecommenderError
    return G