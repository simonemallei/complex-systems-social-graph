import networkx as nx
from scipy.stats import entropy
'''
diversity_of_recommendation returns a metric that represents the entropy of
the opinions of the people in the Recommended List.
The entropy metric is computed by putting each opinion in one of
10 ranges of length 0.2. 
Parameters
----------
  G : {networkx.Graph}
      The graph containing node's opinions.
  n_buckets : {int} default: 10
      The number of buckets used to compute the entropy.
  
Returns
-------
  entropy_dict : {dict}
      The dictionary containing for each graph's node the entropy
      of node's opinions in the list of recommended people
'''
def diversity_of_recommendation(G, n_buckets=10):
    people_recommended_dict = nx.get_node_attributes(G, 'people_recommended')
    opinions = nx.get_node_attributes(G, 'opinion')
    entropy_dict = {}
    for key in people_recommended_dict:
        # Computing entropy for each not empty feed history
        people_recommended = people_recommended_dict[key]
        buckets = [0] * n_buckets
        for person_recommended in people_recommended:
            opinion = opinions[person_recommended]
            # using min in order to not have {n_buckets} as index
            # (in order to have {n_buckets} buckets)
            buck_idx = min(n_buckets - 1, int((opinion + 1.0)* n_buckets / 2))
            # updating counter of {buck_idx}
            buckets[buck_idx] += 1
        # useless if the method is used with the people_recommender method because the latter throws an exception when it cannot recommend anyone
        if len(people_recommended) == 0: 
            entropy_dict[key] = float('nan')
        else:
            buckets = [buck/len(people_recommended) for buck in buckets]
            entropy_dict[key] = entropy(buckets, base = n_buckets)
    return entropy_dict