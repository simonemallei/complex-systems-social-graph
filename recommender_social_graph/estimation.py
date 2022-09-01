import networkx as nx

'''
upd_estim updates the estimation of each node that has posted
its opinion in the last epoch.
It can be updated through "base" strategy or with "kalman" 
strategy (using Kalman Filter).

Parameters
----------
    G : {networkx.Graph}
        The graph containing the social network.
    strategy : {"base", "kalman"} default: "base"
        The string that defines the strategy used by the recommender system.
    strat_param : {dictionary}
        The dictionary containing the parameters value used by the recommender, based
        on {strategy} value.
            - key: "alpha",
              value: alpha coefficient used in "base" strategy.
            - key: "variance",
              value: process variance of "kalman" strategy.
            - key: "variance_measure",
              value: measure variance of "kalman" strategy.
        
Returns
-------
    G : {networkx.Graph}
        The updated graph.
'''
def upd_estim(G, strategy = "base", strat_param = {}):
    # New opinions to estimate (it contains the last content the nodes has posted
    # in the last epoch)
    to_estim = nx.get_node_attributes(G, name='to_estimate')
    # Already estimated opinions by the recommender
    estimated = nx.get_node_attributes(G, name='estimated_opinion')
    if strategy == "base":
        alpha = strat_param.get('alpha', 0.9)
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [])
            estim_op = estimated.get(node_id, 0.0)
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            if not(last_post == []):
                estimated[node_id] = estim_op * alpha + last_post * (1 - alpha)
            to_estim[node_id] = []
            
    elif strategy == "kalman":
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [])
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            if not(last_post == []):
                variance = strat_param.get('variance', 1e-5) # process variance
                R = strat_param.get('variance_measure', 0.1 ** 2) # estimate of measurement variance, change to see effect
                posteri_opinion = nx.get_node_attributes(G, name='posteri_opinion')
                posteri_error = nx.get_node_attributes(G, name='posteri_error')
                # Opinion a posteri (represents the last estimation)
                op_posteri = posteri_opinion.get(node_id, 0.0)
                # Error a posteri (represents the last error value)
                P_posteri = posteri_error.get(node_id, 1.0)

                # Prediction phase
                # Using last posteri values (adding variance to error) as priori in the new epoch
                op_priori = op_posteri
                P_priori = P_posteri + variance

                # Correction phase
                # measurement update
                K = P_priori/(P_priori + R)
                # Compute new opinion and error posteri
                op_posteri = op_priori + K * (last_post - op_priori)
                P_posteri = (1 - K) * P_priori

                # Updating values obtained
                estimated[node_id] = op_posteri
                posteri_opinion[node_id] = op_posteri
                posteri_error[node_id] = P_posteri
                # Updating estimates
                nx.set_node_attributes(G, posteri_opinion, name='posteri_opinion')
                nx.set_node_attributes(G, posteri_error, name='posteri_error')
            
            to_estim[node_id] = []
    nx.set_node_attributes(G, to_estim, name='to_estimate')  
    nx.set_node_attributes(G, estimated, name='estimated_opinion')
    return G

