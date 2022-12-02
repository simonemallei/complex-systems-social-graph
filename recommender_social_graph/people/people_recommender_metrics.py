import networkx as nx
import numpy as np
'''
opinion_estimation_accuracy calculates the accuracy of the various 
opinion estimates. The calculation is done by dividing the absolute distance 
between true and estimated opinion value by 2. The division by 2 (the maximum 
possible value of the absolute distance) results in values 
between 0 and 1.

Parameters
----------
    G : {networkx.Graph}
        The graph containing node's opinions.
  
Returns
-------
    mean, variance : {tuple of floats}
        A tuple containing mean and variance calculated on the 
        every absolute distance.
'''

def opinion_estimation_accuracy(G):
    nodes_with_estimated_opinion = nx.get_node_attributes(G, name='estimated_opinion')
    opinions = nx.get_node_attributes(G, 'opinion')
    estimation_sum_of_distances = []
    for key, value in nodes_with_estimated_opinion.items():
        estimation_sum_of_distances.append(abs(opinions[key] - value) / 2) 
    mean = np.mean(estimation_sum_of_distances)
    variance = np.var(estimation_sum_of_distances)
    return mean, variance

'''
recommendation_homophily_rate calculates the homophily rate 
between the nodes on which the People recommender was used 
and the nodes that were recommended. It is therefore an 
estimate of how well the people recommender is working on 
the basis of the chosen recommendation sub-strategy.
The calculation is done by dividing the absolute distance 
between the opinions by 2. The division by 2 (the maximum 
possible value of the absolute distance) results in values 
between 0 and 1.

Parameters
----------
    G : {networkx.Graph}
        The graph containing node's opinions.
  
Returns
-------
    mean, variance : {tuple of floats}
        A tuple containing mean and variance calculated on the 
        every absolute distance.
'''

def recommendation_homophily_rate(G):
    homophily_rate_list = []
    opinions = nx.get_node_attributes(G, 'opinion')
    person_recommended_dict = nx.get_node_attributes(G, name='person_recommended')
    if person_recommended_dict:
        for key, value in person_recommended_dict.items():
            homophily_rate_list.append(abs(opinions[key] - opinions[value]) / 2)
        mean = np.mean(homophily_rate_list)
        variance = np.var(homophily_rate_list)
        return mean, variance
    else:
        return None, None

    
