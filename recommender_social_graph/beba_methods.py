import networkx as nx
import numpy as np

'''
Original BEBA Model implementation with the new wii parameter.

Parameters
----------
  G         : {networkx.Graph}
      The graph containing the social network.
  beta      : {float}
      Entrenchment parameter of central node
  n_update  : {int}
      index of the node on which to perform the opinion update
  wii       : {int}
      weight noose value on the node on which to perform the opinion update. 
      If this parameter is not populated, the code will calculate wii using 
      the classic formula of the BEBA model

Returns
-------
  G : {networkx.Graph}
      The updated graph.
'''
def compute_update(G, beta, n_update_list, wii = None):
  opinions = nx.get_node_attributes(G, 'opinion')

  for n_update in n_update_list:
    weights = {}

    # Computing weights w(i, j) where i == n_update and (i, j) are edges of G
    # weights[(n_update, n_update)] = beta * opinions[n_update] * opinions[n_update] + 1
    weights[(n_update, n_update)] = wii if wii is not None else beta * opinions[n_update] * opinions[n_update] + 1
    for n_to in G.neighbors(n_update):
      weights[(n_update, n_to)] = beta * opinions[n_update] * opinions[n_to] + 1

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
that function accepts opinion in range [0,1] and scales it in range [-1,1] 
to use it in the beba model

Parameters
----------
  opinion : number 
      opinion to be scaled in range [-1,1]

Returns
-------
  number
      The opinion scaled
'''
def transform_to_BEBA(opinion):
  return 2 * opinion - 1


'''
that function accepts opinion in range [-1,1] and scales it in range [0,1]
to display it in graphics 

Parameters
----------
  opinion : {number} 
      opinion to be scaled in range [0,1]

Returns
-------
          : {number}
      The opinion scaled
'''
def transform_to_scaled(opinion):
  return (opinion + 1) / 2

'''
that function performs pdf test on BEBA model, so it calculates central node's
opinions updates t+1 over different neighbors opinions in a star graph. 
Neighbors opinions are equal to each other and assume values ​​ranging from 1 to 0 
with steps of 0.05. You will notice that w11 weight (noose at the star center) 
is not calculated, but it is a parameter, exactly as it happens in the paper.

Parameters
----------
  num_nodes   : {int}
      Number of nodes in graph (including the central node)
  central_ops : {array of floats}
      array of central node's opinions (range is [0,1])
  beta        : {float}
      Entrenchment parameter of central node
  w11         : {int}
      weight noose value on central node with index 1.

Returns
-------
  x_values_scaled : {array of floats}
      neighbors opinions in the range [0,1]
  z_values_scaled : {array of floats}
      central node opinions calculated at t+1 scaled in range [0,1]
'''
def paper_test(num_nodes, central_ops, beta, w11):
  #NUM_NODES = 5
  #CENTRAL_OPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
  #BETA = 1

  x_values_scaled = [val/10 for val in np.arange(10, -0.5, -0.5)]
  z_values_scaled = []

  # parameter is the neighbors number
  G = nx.star_graph(num_nodes-1)

  for central_op in central_ops:
    z_values_beba = []
    for x_scaled in x_values_scaled:
      neigh_opinions_scaled = [x_scaled]*(num_nodes-1)
      opinions_list_scaled = [central_op, *neigh_opinions_scaled]
      opinions_list_beba = [transform_to_BEBA(item) for item in opinions_list_scaled]
      opinions = dict(zip(G.nodes(), opinions_list_beba))
      nx.set_node_attributes(G, opinions, 'opinion')
      G = compute_update(G, beta, [0], w11)
      opinions = nx.get_node_attributes(G, 'opinion')
      z_values_beba.append(opinions[0])
  
    z_values_scaled.append([transform_to_scaled(item) for item in z_values_beba])
  return x_values_scaled, z_values_scaled