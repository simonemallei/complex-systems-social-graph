from cmath import nan
import networkx as nx
import numpy as np
from tabulate import tabulate
from scipy import spatial
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
import math
from collections import Counter

'''
polarisation returns a metric that represents the polarisation 
of opinions in a graph. 
The polarisation metric is computed  as the sum of the cosine distances
between each opinion vector and the average opinions 

Parameters
----------
  G : {networkx.Graph}
      The graph containing the opinions to measure.
  
Returns
-------
  pol : {float}
      The polarisation metric value
'''
def polarisation(G):
    opinions = list(nx.get_node_attributes(G, 'opinion').values())
    ops = len(opinions[0])
    n = len(opinions)
    means = np.zeros(ops)
    for user in range(n):
        for opinion in range(ops):
            means[opinion] += opinions[user][opinion]
    for opinion in range(ops):
        means[opinion] /= n
    pol = 0
    for user in range(n):
        # Cosine distance between each user's opinion and the mean
        pol += (1 + spatial.distance.cosine(means, opinions[user]))
    return pol


'''
disagreement returns a disagreement metric defined as the
sum of the absolute distance between a node and each 
neighbour, multiplied by the edge's weight.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    dis_dict : {dict}
        The dictionary containing for each graph's node the 
        disagreement value.
'''
def disagreement(G):
    # We need opinions and betas to compute the weights
    opinion = list(nx.get_node_attributes(G, 'opinion').values())
    beta = list(nx.get_node_attributes(G, 'beba_beta').values())
    dis_dict = {}
    for node_from in G.nodes():
        disagreement = 0.0
        # For each node, we compute the disagreement in its neighbourhood
        for node_to in G.neighbors(node_from):
            # Using euclidean distance for measuring distance between opinions 
            weight = beta[node_from] * np.dot(opinion[node_from], opinion[node_to]) / len(opinion[node_from]) + 1
            max_distance = math.sqrt(len(opinion[node_from]) * 4)
            disagreement += np.linalg.norm(opinion[node_from] - opinion[node_to]) / max_distance * weight
        dis_dict[node_from] = disagreement
    return dis_dict
    
    
'''
sarle_bimodality returns a metric that tests bimodality.
The bimodality coefficient is computed as: (skewness ^ 2 + 1) / (kurtosis)
We compute sarle_bimodality for each dimension
Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    bimodality : {list of float}
        The Sarle's bimodality metric value for each dimension.
'''
def sarle_bimodality(G, ops):
    opinions = list(nx.get_node_attributes(G, 'opinion').values())
    bimodality = []
    for opinion in range(ops):
        current = np.zeros(len(opinions))
        for user in range(len(opinions)):
            current[user] = opinions[user][opinion]
        bimodality.append(((skew(current) ** 2) + 1) / kurtosis(current))
    return bimodality



'''
feed_entropy returns a metric that represents the entropy of
the feed history.
The entropy metric is computed by putting each content in one of
10 ranges of length 0.2 based on its opinion. 

Parameters
----------
    G : {networkx.Graph}
        The graph containing the feed history to measure.
    n_buckets : {int}, default : 4
        The number of buckets for each opinion used to compute the entropy.
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
def feed_entropy(G, ops, n_buckets=4, max_len_history=30):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    entropy_dict = {}
    for node in G.nodes():
        # Computing entropy for each non-empty feed history
        curr_history = feed_history.get(node, [[] for i in range(ops)])
        entropy_sum = 0.0
        cnt = 0
        for op in range(ops):
            if len(curr_history[op]) != 0:
                cnt += 1
                if len(curr_history[op]) > max_len_history:
                    curr_history[op] = curr_history[op][-max_len_history:]
                buckets = [0] * n_buckets
                for content in curr_history[op]:
                    buck_idx = min(n_buckets - 1, int((content + 1.0)* n_buckets / 2))
                    buckets[buck_idx] += 1
                buckets = [buck/len(curr_history[op]) for buck in buckets]
                entropy_sum  += entropy(buckets, base = n_buckets)
        if cnt > 0:
            entropy_dict[node] = entropy_sum / cnt
        else:
            entropy_dict[node] = nan
    return entropy_dict