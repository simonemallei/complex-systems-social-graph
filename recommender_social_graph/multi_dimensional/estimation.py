import networkx as nx

def upd_estim(G, ops, strategy = "base", strat_param = {}):
    # New opinions to estimate (it contains the last content the nodes has posted
    # in the last epoch)
    to_estim = nx.get_node_attributes(G, name='to_estimate')
    # Already estimated opinions by the recommender
    # estimated_opinion is a dictionary which contains, for each node, a list of the estimated opinions
    estimated = nx.get_node_attributes(G, name='estimated_opinion')
    if strategy == "base":
        alpha = strat_param.get('alpha', 0.9)
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [[] for i in range(ops)])
            estim_op = estimated.get(node_id, [0.0] * ops)
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            for op in range(ops):
                if not(last_post[op] == []):
                    estimated[node_id][op] = estim_op[op] * alpha + last_post[op] * (1 - alpha)
                to_estim[node_id][op] = []
    elif strategy == "kalman":
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [[] for i in range(ops)])
            for op in range(ops):
                if not(last_post[op] == []):
                    variance = strat_param.get('variance', 1e-5)
                    R = strat_param.get('variance_measure', 0.1 ** 2)
                    posteri_opinion = nx.get_node_attributes(G, name='posteri_opinion')
                    posteri_error = nx.get_node_attributes(G, name='posteri_error')
                    op_posteri = posteri_opinion.get(node_id, [0.0] * ops)
                    P_posteri = posteri_error.get(node_id, [0.0] * ops)
                    # Prediction phase
                    op_priori = op_posteri[op]
                    P_priori = P_posteri[op] + variance
                    # Correction phase
                    K = P_priori / (P_priori + R)
                    op_posteri[op] = op_priori + K * (last_post[op] - op_priori)
                    P_posteri[op] = (1 - K) * P_priori
                    # Updating values obtained
                    estimated[node_id][op] = op_posteri[op]
                    posteri_opinion[node_id][op] = op_posteri[op]
                    posteri_error[node_id][op] = P_posteri[op]
                    # Updating estimates
                    nx.set_node_attributes(G, posteri_opinion, name='posteri_opinion')
                    nx.set_node_attributes(G, posteri_error, name='posteri_error')
                to_estim[node_id][op] = []
    # Updating estimated opinions
    nx.set_node_attributes(G, to_estim, name='to_estimate')  
    nx.set_node_attributes(G, estimated, name='estimated_opinion')
    return G
