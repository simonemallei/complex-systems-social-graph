import networkx as nx
import numpy as np
from tabulate import tabulate
from scipy import spatial

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
        pol += spatial.distance.cosine(means, opinions[user])
    return pol
