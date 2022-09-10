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
            weight = beta[node_from] * opinion[node_from] * opinion[node_to] + 1
            #meglio fare la media piuttosto che sommare. Il peso dell'arco va tolto perché va controcorrente alla distanza assoluta
            disagreement += abs(opinion[node_from] - opinion[node_to]) * weight
        dis_dict[node_from] = disagreement
    return dis_dict