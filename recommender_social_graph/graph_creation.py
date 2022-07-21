import networkx as nx
import numpy as np
from collections import defaultdict
import random

def MY_homophilic_ba_graph(N, m,alpha=2, beta=1):
    """Return homophilic random graph using BA preferential attachment model.
    A graph of n nodes is grown by attaching new nodes each with m
    edges that are preferentially attached to existing nodes with high
    degree. The connections are established by linking probability which 
    depends on the connectivity of sites and the homophily(similarities).
    homophily varies ranges from 0 to 1.

    Parameters
    ----------
    N : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
   """

    G = nx.Graph()
    node_attribute = {}
    

    for n in range(N):
        #generate opinion in range [-1, 1]
        op = random.random() * 2 - 1
        G.add_node(n , opinion = op)
        node_attribute[n] = op

    #create homophilic distance ### faster to do it outside loop ###
    dist = defaultdict(float) #distance between nodes

    for n1 in range(N):
        n1_attr = node_attribute[n1]
        for n2 in range(N):
            n2_attr = node_attribute[n2]
            dist[(n1,n2)] = abs(n1_attr - n2_attr)


    target_list = list(range(m))
    source = m #start with m nodes

    while source < N:
        targets = _pick_targets(G,source,target_list,dist,m, alpha=alpha ,beta=beta)
        if targets != set(): #if the node does  find the neighbor
            G.add_edges_from(zip([source]*m,targets))

        target_list.append(source)  #target list is updated with all the nodes in the graph 
        source += 1

    return G

def _pick_targets(G,source,target_list,dist,m ,alpha, beta):
    '''
    First compute the target_prob which is related to the degree'''
    target_prob_dict = {}
    for target in target_list:
        pow_dist =  (dist[(source,target)]+1)**alpha
        target_prob = (1/pow_dist)*((G.degree(target)+0.00001)**beta) #formula to compute targer prob, >>Degree better chance
        target_prob_dict[target] = target_prob
        
    prob_sum = sum(target_prob_dict.values())

    targets = set()
    target_list_copy = target_list.copy()
    count_looking = 0
    if prob_sum == 0:
        return targets #it returns an empty set

    while len(targets) < m:
        count_looking += 1
        if count_looking > len(G): # if node fails to find target
            break
        rand_num = random.random()
        cumsum = 0.0
        for k in target_list_copy:
            cumsum += float(target_prob_dict[k]) / prob_sum
            if rand_num < cumsum: 
                targets.add(k)
                target_list_copy.remove(k)
                break
    return targets

'''
create_graph creates a graph using the method MY_homophilic_ba_graph,
then it adds the beta used for BEBA and ABEBA models.

Parameters
----------
  n_ag : {int}
      The numbers of nodes in the graph.
  beba_beta : {list of float}
      The list with nodes' beta values.
  avg_friend: {}
      The number of average neighbors for each node in the graph.
  hp_alpha : {float}
      MY_homophilic_graph's homophily parameter.
  hp_beta : {float}
      MY_homophilic_graph's preferential attachment parameter.

Returns
-------
  G : {networkx.Graph}
      Returns the graph obtained.
'''
def create_graph(n_ag, beba_beta=[1] , avg_friend=3, hp_alpha=2, hp_beta=1):
  
  # checks on beba_beta length
  if len(beba_beta) != 1 and len(beba_beta) != n_ag:
    print("WARNING: beba_beta length is not valid. It must be 1 or nodes' number. Default value will be used")
    beba_beta = [1] * n_ag

  if len(beba_beta) == 1:
    beba_beta = [beba_beta[0]] * n_ag

  # Calls MY_homophilic_ba_graph
  G = MY_homophilic_ba_graph(n_ag, avg_friend, hp_alpha, hp_beta)

  # Setting beba_beta as node attributes
  node_beba_beta_dict = dict(zip(G.nodes(), beba_beta))
  nx.set_node_attributes(G, node_beba_beta_dict, 'beba_beta')

  return G

