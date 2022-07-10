import networkx as nx
import numpy as np

def people_recommender(G, act_nodes, strategy="random"):
    nodes = list(G.nodes)
    for node_id in act_nodes:
        neigs = list(nx.neighbors(G, node_id))
        recommended_friends = [x for x in nodes if x not in neigs]
        recommended_friends.remove(node_id)
        if strategy == "random":
            # recommending a random node not already friend as a new friend
            recommended_friend = np.random.choice(range(len(recommended_friends)), size=1, replace=False)
            G.add_edge(node_id, recommended_friend)
            # deleting a random edge to prevent fully connected graphs
            discarded_friend = np.random.choice(range(len(neigs)), size=1, replace=False)
            G.remove_edge(node_id, discarded_friend)

    return G