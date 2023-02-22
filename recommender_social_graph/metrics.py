import networkx as nx
import numpy as np
from scipy.stats import skew, kurtosis, entropy

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
    # We need opinions to compute the weights
    opinions_dict = nx.get_node_attributes(G, 'opinion')
    dis_dict = {}
    for node_from in G.nodes():
        # For each node, we compute the disagreement in its neighbourhood
        disagreement = [abs(opinions_dict[node_from] - opinions_dict[node_to]) for node_to in G.neighbors(node_from)]
        # It prevents mean and variance on empty disagreements (nodes without friends)
        if disagreement != []:
            dis_dict[node_from] = (np.mean(disagreement), np.var(disagreement))
        
    means = [dis_dict[node][0] for node in dis_dict.keys()]
    variances = [dis_dict[node][1] for node in dis_dict.keys()]
    mean = np.mean(means)
    variance = np.sum(variances) / (len(G.nodes()) ** 2)
    return mean, variance

'''
echo_chamber_value returns a value representing how far the given node 
is inside an echo chamber. 
For each node k with neighbors, this value is found by calculating the 
Kullback-Leibler Divergence between the probability distribution on the opinions 
of the k-neighbors, and the probability distribution on the opinions of 
the nodes in the entire graph. Each probability distribution is calculated using 
numpy's histogram method with 20 bins.

Parameters
----------
    G : {networkx.Graph}
        The graph containing the opinions to measure.
  
Returns
-------
    echo_chamber_dict, mean : {tuple}
        first element: a dictionary that has as keys the nodes on which it was 
        possible to calculate the metric (therefore non-isolated nodes), and 
        as values the result of the Kullback-Leibler Divergence.
        second element: mean performed over all dictionary values
'''
def echo_chamber_value(G):
    opinions_list = list(nx.get_node_attributes(G, 'opinion').values())
    opinions_pdf, _ = np.histogram(opinions_list, bins=20, density=True)
    opinions_dict = nx.get_node_attributes(G, 'opinion')
    echo_chamber_dict = {}
    for node in G.nodes():
        neighbors_list = list(G.neighbors(node))
        if len(neighbors_list) > 0:
            neigh_opinions_list = [opinions_dict[neigh] for neigh in neighbors_list]
            neigh_opinions_pdf, _ = np.histogram(neigh_opinions_list, bins=20, density=True)
            echo_chamber_dict[node] = entropy(neigh_opinions_pdf, qk=opinions_pdf)
            
    mean = np.mean(list(echo_chamber_dict.values()))
    # Managing infinite values to obtaint valid json values
    echo_chamber_dict_sanitized = {key : str(value) if np.isinf(value) else value for key, value in echo_chamber_dict.items()}
    return echo_chamber_dict_sanitized, str(mean) if np.isinf(mean) else mean
