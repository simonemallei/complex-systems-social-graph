import networkx as nx
import numpy as np
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
      The polarisation metric value
'''
def polarisation(G):
  opinions = list(nx.get_node_attributes(G, 'opinion').values())
  np_op = np.array(opinions)
  mean_op = np.mean(np_op)
  # Computing polarisation
  pol = np.sum((np_op - mean_op) * (np_op - mean_op))
  
  return pol
