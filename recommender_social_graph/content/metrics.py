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
feed_satisfaction returns a satisfaction metric that uses the Cobb-Douglas
utility for each node using cd_beta = 1 / (2 ** (beta[node] + 1)).
The Cobb-Douglas utility is then normalized using the length of
the history feed considered, obtaining the satisfaction value.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the feed history to measure.
    max_len_history : {int}, default : 20
        Maximum length of the feed history considered (if we have
        more than {max_len_history} posts, we'll consider
        the {max_len_history} newest ones).
  
Returns
-------
    sat_dict : {dict}
        The dictionary containing for each graph's node the 
        satisfaction value of its feed history.
'''
def feed_satisfaction(G, max_len_history = 20):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    beta = nx.get_node_attributes(G, 'beba_beta')
    opinion = nx.get_node_attributes(G, 'opinion')
    sat_dict = {}
    for node in G.nodes():
        # Computing normalized Cobb-Douglas Utility for each non-empty feed history
        curr_history = np.array(feed_history.get(node, []))
        # multiplying each content's opinion for the node's sign
        # so if a value in the history is > 0, then it has the same sign of the
        # node's opinion 
        sign_node = opinion[node] / abs(opinion[node])
        curr_history = curr_history * sign_node
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
            is_positive = np.where(curr_history > 0.0, 1, 0)
            # count the elements if a positive sign, 
            # i.e. the number of posts with the same sign of the
            # node's opinion
            cnt_positive = np.sum(is_positive)
            # Cobb-Douglas beta decreases exponentially by
            # increasing the node's beta value (we'll be less
            # satisfied by the opposed content if we have
            # an higher bias, due to the backfire effect)
            cd_beta = 1 / (2 ** (beta[node] + 1))
            cd_alpha = 1 - cd_beta
            # applying Cobb-Douglas Utility formula (if one of the elements
            # has a value of 0, it will be swapped with 1
            utility = ((max(1, cnt_positive) ** cd_alpha) * 
                       (max(1, (len_feed - cnt_positive)) ** cd_beta))
            # computing maximum possible satisfaction with {len_feed + 1}
            # elements in order to normalize the satisfaction obtained
            max_x = (len_feed + 1) * cd_alpha
            max_sat = (max_x ** cd_alpha) * (((len_feed + 1) - max_x) ** cd_beta) 
            sat_dict[node] = utility / max_sat
    return sat_dict

'''
feed_satisfaction_weight returns a satisfaction metric that uses the Cobb-Douglas
utility for each node using cd_beta = 1 / (2 ** (beta[node] + 1)).
The Cobb-Douglas utility is then normalized using the sum of the absolute values
of the history feed considered, obtaining the satisfaction value.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the feed history to measure.
    max_len_history : {int}, default : 20
        Maximum length of the feed history considered (if we have
        more than {max_len_history} posts, we'll consider
        the {max_len_history} newest ones).
  
Returns
-------
    sat_dict : {dict}
        The dictionary containing for each graph's node the 
        satisfaction value of its feed history.
'''
def feed_satisfaction_weight(G, max_len_history = 20):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    beta = nx.get_node_attributes(G, 'beba_beta')
    opinion = nx.get_node_attributes(G, 'opinion')
    sat_dict = {}
    for node in G.nodes():
        # Computing normalized Cobb-Douglas Utility for each non-empty feed history
        curr_history = np.array(feed_history.get(node, []))
        # multiplying each content's opinion for the node's sign
        # so if a value in the history is > 0, then it has the same sign of the
        # node's opinion 
        sign_node = opinion[node] / abs(opinion[node])
        curr_history = curr_history * sign_node
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
            pos_arr = np.where(curr_history > 0.0, curr_history, 0)
            neg_arr = np.where(curr_history < 0.0, -curr_history, 0)
            # sum the absolute values of posts with the same sign
            # of the node's opinion in {pos_sum}, the opposite ones
            # in {neg_sum}
            pos_sum = 1 + np.sum(pos_arr)
            neg_sum = 1 + np.sum(neg_arr)
            # Cobb-Douglas beta decreases exponentially by
            # increasing the node's beta value (we'll be less
            # satisfied by the opposed content if we have
            # an higher bias, due to the backfire effect)
            cd_beta = 1 / (2 ** (beta[node] + 1))
            cd_alpha = 1 - cd_beta
            # applying Cobb-Douglas Utility formula
            utility = ((pos_sum ** cd_alpha) * 
                       (neg_sum ** cd_beta))
            # computing maximum possible satisfaction with {pos_sum + neg_sum}
            # in order to normalize the satisfaction obtained
            max_x = (pos_sum + neg_sum) * cd_alpha
            max_sat = (max_x ** cd_alpha) * (((pos_sum + neg_sum) - max_x) ** cd_beta) 
            sat_dict[node] = utility / max_sat
    return sat_dict