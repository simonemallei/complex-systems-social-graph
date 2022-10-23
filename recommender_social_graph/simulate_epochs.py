import numpy as np
import networkx as nx
from abeba_methods import compute_activation, compute_post
from content.content_recommender import monitor_feed, content_recommender
from estimation import upd_estim
from people.people_recommender import people_recommender, RecommendedFriendNotFound, StrategyNotRecognized, SubstrategyNotRecognized
from metrics import polarisation, sarle_bimodality, disagreement
from content.metrics import feed_entropy, feed_satisfaction
from people.people_recommender_metrics import opinion_estimation_accuracy, recommendation_homophily_rate

class SimulateEpochContentPeopleRecommenderError(Exception):
    """Raised when an error occurred in the simulate_epoch_content_people_recommender method"""
    pass

class simulateEpochsError(Exception):
    """Raised when an error occurred in the simulate_epochs method"""
    pass

'''
compute_metrics calculates the metrics in the current epoch.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.

Returns
-------
    epoch_metrics : {dictionary}
        It contains values of all metrics calculated in the current epoch.
'''

def compute_metrics(G):
    epoch_metrics = {}
    epoch_metrics["polarization_value"] = polarisation(G)
    epoch_metrics["bimodality"] = sarle_bimodality(G)

    disagreement_result = {}
    disagreement_result["mean"], disagreement_result["variance"] = disagreement(G)
    epoch_metrics["disagreement"] = disagreement_result

    feed_entropy_result = {}
    feed_entropy_result["mean"], feed_entropy_result["variance"] = feed_entropy(G, n_buckets=10, max_len_history=30)
    epoch_metrics["feed_entropy"] = feed_entropy_result

    feed_satisfaction_result = {}
    feed_satisfaction_result["mean"], feed_satisfaction_result["variance"] = feed_satisfaction(G, max_len_history = 10, sat_alpha = 0.75)
    epoch_metrics["feed_satisfaction"] = feed_satisfaction_result

    opinion_estimation_accuracy_result = {}
    opinion_estimation_accuracy_result["mean"], opinion_estimation_accuracy_result["variance"] = opinion_estimation_accuracy(G)
    epoch_metrics["opinion_estimation_accuracy"] = opinion_estimation_accuracy_result

    recommendation_homophily_rate_result = {}
    recommendation_homophily_rate_result["mean"], recommendation_homophily_rate_result["variance"] = recommendation_homophily_rate(G)
    epoch_metrics["recommendation_homophily_rate"] = recommendation_homophily_rate_result

    return epoch_metrics

'''
simulate_epoch_content_people_recommender simulates an epoch. It randomly activates a 
subset ({rate_updating_nodes * len(G.nodes())}) of graph's nodes: firstly the content
recommender will update their feed, then each node will update its opinion 
base on its feed.
Afterwards, a percentage of the activated nodes (always sampled randomly) will 
also be posting nodes, updating their neighbours' feed with the content. The 
opinion shared by the posting nodes has a noise related to the parameter {epsilon}.
Then the people recommender is run on the posting_nodes.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    rate_updating_nodes : {float}
        The percentage of the nodes that will be activated. Interval [0,1]
    epsilon : {float} default: 0.0
        The Gaussian noise's standard deviation in the posting phase.
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
    strategy_content_recommender : {"random", "normal", "nudge", "nudge_opt", "similar", "unsimilar"} default: "random"
        The string that defines the strategy used by the recommender system.
    strat_param_content_recommender : {dictionary}
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
def simulate_epoch_content_people_recommender(
        G, 
        rate_updating_nodes, 
        epsilon = 0.0, 
        estim_strategy = "base", 
        estim_strategy_param = {},
        strategy_content_recommender = "random",
        strat_param_content_recommender = {},
        strategy_people_recommender = "random",
        substrategy_people_recommender = None
    ):

    # Sampling randomly the activating nodes
    updating_nodes = int(rate_updating_nodes * len(G.nodes()))
    act_nodes = np.random.choice(range(len(G.nodes())), size=updating_nodes, replace=False)

    # Executing content recommender system on activated nodes
    G = content_recommender(G, act_nodes, strategy_content_recommender, strat_param_content_recommender)
    # Monitoring feeds that are going to be cleared 
    G = monitor_feed(G, act_nodes)
    # Executing activation phase: activated nodes will consume their feed
    G = compute_activation(G, act_nodes)

    # Executing posting phase: activated nodes will post in their neighbours' feed
    G, posting_nodes_list = compute_post(G, act_nodes, epsilon)
    # Estimating opinion by the recommender
    G = upd_estim(G, posting_nodes_list, strategy = estim_strategy, strat_param = estim_strategy_param)
    try:
      G = people_recommender(G, posting_nodes_list, strategy_people_recommender, substrategy_people_recommender)
    except (RecommendedFriendNotFound, StrategyNotRecognized, SubstrategyNotRecognized):
      print("the People Recommender failed to recommend a new friend to a given node")
      raise SimulateEpochContentPeopleRecommenderError
    return G

'''
simulate_epochs first saves the initial opinions. Then a cycle is made on each epoch, 
in which the simulate_epoch_content_people_recommender method is called first, which 
simulates a single epoch, and then the compute_metrics method, which calculates the 
metrics on the current epoch.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    model_params: {dict}
        It contains the following parameters:
            rate_updating_nodes : {float}
                The percentage of the nodes that will be activated. Interval [0,1]
            epsilon : {float} default: 0.0
                The Gaussian noise's standard deviation in the posting phase.
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
            strategy_content_recommender : {"random", "normal", "nudge", "nudge_opt", "similar", "unsimilar"} default: "random"
                The string that defines the strategy used by the recommender system.
            strat_param_content_recommender : {dictionary}
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
            strategy_people_recommender : {String} default: "random"
                The string that defines the strategy used by the recommender system.
                There are two possible strategies that can be combined with two possible sub-strategies,
                in addition to the random strategy:
                Strategies: opinion_estimation_based, topology_based
            Substrategies: {String} default: None
                Possible values are: counteract_homophily, favour_homophily

Returns
-------
    G, initial_opinions, opinions_and_metrics : {tuple}
        A tuple containing the final graph, the initial views, and the views and metrics for each epoch
'''
def simulate_epochs(G, model_params):
    opinions_and_metrics = []
    initial_opinions = list(nx.get_node_attributes(G, 'opinion'))
    for i in range(model_params["num_epochs"]):
        epoch_data = {}
        try:
            G = simulate_epoch_content_people_recommender(
                        G = G,
                        rate_updating_nodes = model_params["rate_updating_nodes"],
                        epsilon = model_params["post_epsilon"],
                        estim_strategy = model_params["estim_strategy"],
                        estim_strategy_param = model_params["estim_strategy_param"],
                        strategy_content_recommender = model_params["strategy_content_recommender"],
                        strat_param_content_recommender = model_params["strat_param_content_recommender"], 
                        strategy_people_recommender = model_params["strategy_people_recommender"],
                        substrategy_people_recommender = model_params["substrategy_people_recommender"]
                    )
        except SimulateEpochContentPeopleRecommenderError:
            print("An error occurred in the simulate_epoch_content_people_recommender method")
            raise simulateEpochsError
        epoch_metrics = compute_metrics(G)
        epoch_data["opinions"] = list(nx.get_node_attributes(G, 'opinion'))
        epoch_data["metrics"] = epoch_metrics
        opinions_and_metrics.append(epoch_data)
    return G, initial_opinions, opinions_and_metrics
              
