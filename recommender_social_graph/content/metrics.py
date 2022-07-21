import scipy.stats.entropy

'''
feed_entropy returns a metric that represents the entropy of
the feed history.
The entropy metric is computed by putting each content in one of
10 ranges of length 0.2 based on its opinion. 

Parameters
----------
  G : {networkx.Graph}
      The graph containing the feed history to measure.
  n_buckets : {int} default: 10
      The number of buckets used to compute the entropy.
  
Returns
-------
  entropy : {dict}
      The dictionary containing for each graph's node the entropy
      of its feed history.
'''
def feed_entropy(G, n_buckets=10):
    feed_history = nx.get_node_attributes(G, 'feed_history')
    entropy = {}
    for node in G.nodes():
        # Computing entropy for each not empty feed history
        curr_history = feed_history.get(node, [])
        buckets = [0] * n_buckets
        for content in curr_history:
            # using min in order to not have {n_buckets} as index
            # (in order to have {n_buckets} buckets)
            buck_idx = min(n_buckets - 1, int((content + 1.0)* n_buckets / 2))
            # updating counter of {buck_idx}
            buckets[buck_idx] += 1
        count_feed = sum(buckets)
        buckets = [buck/count_feed for buck in buckets]
        if len(curr_history) != 0:
            entropy[node] = scipy.stats.entropy(buckets, base = n_buckets)
    return entropy
