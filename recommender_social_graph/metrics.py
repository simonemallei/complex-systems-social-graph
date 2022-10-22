import networkx as nx
import numpy as np
from scipy.stats import skew, kurtosis
'''
polarisation returns a metric that represents the polarisation 
of opinions in a graph. 
The polarisation metric is computed  as the sum of squared 
difference between each user and mean opinion.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    pol : {float}
        The polarisation metric value.
'''
def polarisation(G):
    opinions = list(nx.get_node_attributes(G, 'opinion').values())
    np_op = np.array(opinions)
    mean_op = np.mean(np_op)
    # Computing polarisation
    pol = np.sum((np_op - mean_op) * (np_op - mean_op))
  
    return pol

'''
sarle_bimodality returns a metric that tests bimodality.
The bimodality coefficient is computed as: (skewness ^ 2 + 1) / (kurtosis)

Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    bimodality : {float}
        The Sarle's bimodality metric value.
'''
def sarle_bimodality(G):
    opinions = list(nx.get_node_attributes(G, 'opinion').values())
    np_op = np.array(opinions)
    bimodality = ((skew(np_op) ** 2) + 1) / kurtosis(np_op)
    return bimodality

'''
disagreement returns a disagreement metric defined as the
distribution of the absolute distance between a node and each 
neighbour.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    mean, variance : {tuple of floats}
        A tuple containing mean and variance calculated on the 
        disagreement mean and variance for each graph's node
'''
def disagreement(G):
    # We need opinions and betas to compute the weights
    opinion = list(nx.get_node_attributes(G, 'opinion').values())
    dis_dict = {}
    for node_from in G.nodes():
        # For each node, we compute the disagreement in its neighbourhood
        disagreement = [abs(opinion[node_from] - opinion[node_to]) for node_to in G.neighbors(node_from)]
        dis_dict[node_from] = (np.mean(disagreement), np.var(disagreement))

    means = [dis_dict[node][0] for node in G.nodes()]
    variances = [dis_dict[node][1] for node in G.nodes()]
    mean = np.mean(means)
    variance = np.sum(variances) / (len(G.nodes()) ** 2)
    return mean, variance