import networkx as nx
from scipy.stats import entropy
import numpy as np
'''
feed_entropy returns a metric that represents the entropy of
the feed history.
The entropy metric is computed by putting each content in one of
10 ranges of length 0.2 based on its opinion. 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the feed history to measure.
    n_buckets : {int}, default : 10
        The number of buckets used to compute the entropy.
    max_len_history : {int}, default : 30
        Maximum length of the feed history considered (if we have
        more than {max_len_history} posts, we'll consider
        the {max_len_history} newest ones).
  
Returns
-------
    entropy_dict : {dict}
        The dictionary containing for each graph's node the entropy
        of its feed history.
'''
def feed_entropy(G, n_buckets=10, max_len_history=30):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    entropy_dict = {}
    for node in G.nodes():
        # Computing entropy for each non-empty feed history
        curr_history = feed_history.get(node, [])
        len_feed = len(curr_history)
        # if the feed history is empty, the entropy has NaN value
        if len_feed == 0:
            entropy_dict[node] = float('nan')
        else:
            # if the history's length is greater than the maximum,
            # we'll consider the {max_len_history} newest ones
            if len_feed >= max_len_history:
                curr_history = curr_history[-max_len_history:]
                len_feed = max_len_history
            buckets = [0] * n_buckets
            for content in curr_history:
                # using min in order to not have {n_buckets} as index
                # ({n_buckets} possible buckets)
                buck_idx = min(n_buckets - 1, int((content + 1.0)* n_buckets / 2))
                # updating counter of {buck_idx}
                buckets[buck_idx] += 1
            # defining the buckets as a probability distribution 
            # in order of being able to compute its entropy
            buckets = [buck/len_feed for buck in buckets]
            entropy_dict[node] = entropy(buckets, base = n_buckets)
    return entropy_dict


'''
feed_satisfaction returns a satisfaction metric that uses the ABEBA weight 
between a node and a post in its feed, the node's opinion and its bias (beta). 
It is defined as {weight} / (1 + {beta[node]} * {opinion[node]}).

Parameters
----------
    G : {networkx.Graph}
        The graph containing the feed history to measure.
    max_len_history : {int}, default : 10
        Maximum length of the feed history considered (if we have
        more than {max_len_history} posts, we'll consider
        the {max_len_history} newest ones).
    sat_alpha : {float}, default : 0.75
        Alpha coefficient used to weight previous satisfaction
        of the node to compute the current one.
  
Returns
-------
    sat_dict : {dict}
        The dictionary containing for each graph's node the 
        satisfaction value of its feed history.
'''
def feed_satisfaction(G, max_len_history = 10, sat_alpha = 0.75):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    beta = nx.get_node_attributes(G, 'beba_beta')
    opinion = nx.get_node_attributes(G, 'opinion')
    sat_dict = {}
    for node in G.nodes():
        curr_history = np.array(feed_history.get(node, []))
        len_feed = len(curr_history)
        # if the feed history is empty, the satisfaction has NaN value
        if len_feed == 0:
            sat_dict[node] = float('nan')
        else:
            # if the history's length is greater than the maximum,
            # we'll consider the {max_len_history} newest ones
            if len_feed >= max_len_history:
                curr_history = curr_history[-max_len_history:]
                len_feed = max_len_history
            # computing weights of each content and then the function is defined as:
            # f(weight) = (weights) / (1 + beta[node] * opinion[node])
            # afterwards, we compute satisfaction as:
            # sat[node] = satisf * sat_alpha + (1 - sat_alpha) * np.mean(sig_x)
            weights = curr_history * opinion[node] * beta[node] + 1
            sig_x = (weights) / (1 + beta[node] * opinion[node])
            satisf = sat_dict.get(node, np.mean(sig_x))
            sat_dict[node] = satisf * sat_alpha + (1 - sat_alpha) * np.mean(sig_x)

    return sat_dict
    