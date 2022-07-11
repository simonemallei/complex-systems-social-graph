import networkx as nx
import numpy as np

class LengthIsNotEqualError(Exception):
    """Raised when an input list is not the same size as the number of nodes to update"""
    pass

class ComputeUpdateError(Exception):
    """Raised when an error occurred in the compute_update method"""
    pass


'''
Original BEBA Model implementation with the new wii parameter.

Parameters
----------
  G         : {networkx.Graph}
      The graph containing the social network.
  beta_list      : {list of floats}
      Entrenchment parameters of updating nodes.
      The index of the list corresponds to the index of the node.
  n_update  : {list of int}
      indexes list of the nodes on which to perform the opinion update
  wii       : {list of floats}
      weight noose value on the nodes on which to perform the opinion update.     
      The index of the list corresponds to the index of the node.
      If this parameter is not populated, the code will calculate wii using 
      the classic formula of the BEBA model

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def compute_update(G, beta_list, n_update_list, wii_list = None):
  if len(n_update_list) is not len(beta_list) or (wii_list is not None and len(n_update_list) is not len(wii_list)):
    raise LengthIsNotEqualError

  opinions = nx.get_node_attributes(G, 'opinion')

  for idx, n_update in enumerate(n_update_list):
    weights = {}

    # Computing weights w(i, j) where i == n_update and (i, j) are edges of G
    # weights[(n_update, n_update)] = beta * opinions[n_update] * opinions[n_update] + 1
    weights[(n_update, n_update)] = wii_list[idx] if wii_list is not None else beta_list[idx] * opinions[n_update] * opinions[n_update] + 1
    for n_to in G.neighbors(n_update):
      weights[(n_update, n_to)] = beta_list[idx] * opinions[n_update] * opinions[n_to] + 1

    # Computing new opinion of n_update
    op_num = weights[(n_update, n_update)] * opinions[n_update]
    op_den = weights[(n_update, n_update)]
    for n_to in G.neighbors(n_update):
      op_num += weights[(n_update, n_to)] * opinions[n_to]
      op_den += weights[(n_update, n_to)]

    # If the denominator is < 0, the opinion gets polarized and 
    # the value is set to sgn(opinions[n_update])
    if op_den <= 0:
      opinions[n_update] = opinions[n_update] / abs(opinions[n_update])
    else:
      opinions[n_update] = op_num / op_den
    
    # Opinions are capped within [-1, 1] 
    if opinions[n_update] < -1:
      opinions[n_update] = -1
    if opinions[n_update] > 1:
      opinions[n_update] = 1

    nx.set_node_attributes(G, opinions, 'opinion')

  return G

'''
Original BEBA Model implementation with the new wii parameter.

Parameters
----------
  G         : {networkx.Graph}
      The graph containing the social network.
  beta      : {float}
      Entrenchment parameter of central node
  n_update  : {list of int}
      indexes list of the nodes on which to perform the opinion update
  wii       : {list of floats}
      weight noose value on the nodes on which to perform the opinion update.     
      The index of the list corresponds to the index of the node.
      If this parameter is not populated, the code will calculate wii using 
      the classic formula of the BEBA model

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def simulate_epoch_updated(G, beta, percent_updating_nodes, wii_list = None):
  # Sampling randomly the updating nodes
  n_updating_nodes = int(percent_updating_nodes * len(G.nodes()) / 100)
  updating_nodes = np.random.choice(range(len(G.nodes())), size=n_updating_nodes, replace=False)
  beta_list = [beta] * n_updating_nodes
  try:
    G = compute_update(G, beta_list, updating_nodes, wii_list)
  except LengthIsNotEqualError:
    print("beta_list and/or wii list do not have the same length as the list of nodes to be updated")
    raise ComputeUpdateError
  return G
