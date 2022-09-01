import networkx as nx
import math

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
    elif strategy == "kalman_beta":
        posteri_opinion = nx.get_node_attributes(G, name='posteri_opinion')
        posteri_error = nx.get_node_attributes(G, name='posteri_error')
        estim_epoch = nx.get_node_attributes(G, name='estim_epoch')
        feed_history = nx.get_node_attributes(G, name='feed_history')
        feed_epoch = nx.get_node_attributes(G, name='feed_epoch') 
        estimated_beta = nx.get_node_attributes(G, name='estimated_beta')
        var_beta = nx.get_node_attributes(G, name='var_beta')
               
        for node_id in G.nodes():
            last_post = to_estim.get(node_id, [])
            # If last_post is == [], then the node hasn't posted anything
            # in the last epoch.
            if not(last_post == []):
                variance = strat_param.get('variance', 1e-5) # process variance
                R = strat_param.get('variance_measure', 0.1 ** 2) # estimate of measurement variance, change to see effect
                # Opinion a posteri (represents the last estimation)
                op_posteri = posteri_opinion.get(node_id, 0.0)
                prev_op = op_posteri
                # Error a posteri (represents the last error value)
                P_posteri = posteri_error.get(node_id, 1.0)
                prev_sd = math.sqrt(P_posteri)

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

                # Updating beta estimation
                curr_feed_history = feed_history.get(node_id, [])
                curr_feed_epoch = feed_epoch.get(node_id, [])
                if (P_posteri < 7e-3):
                    # Considering left extreme of the 95% confidence interval from the previous estimation
                    left_prev_op = prev_op - 2 * prev_sd
                    left_beta = 0.0
                    
                    next_epoch = estim_epoch.get(node_id, 0)
                    
                    if [x for x in curr_feed_epoch if next_epoch <= x] == []:
                        break

                    # computing beta and constant coefficients in the numerator
                    # of the fraction obtained by computing
                    # ABEBA(yi(t+1)) - yi(t+1) = 0 
                    beta_coeff = left_prev_op ** 3
                    beta_coeff -= (left_prev_op ** 2) * op_posteri
                    const_coeff = -left_prev_op
                    const_coeff += op_posteri
                    for post_idx in range(len(curr_feed_epoch)):
                        if curr_feed_epoch[post_idx] >= next_epoch:
                            beta_coeff += left_prev_op * (curr_feed_history[post_idx] ** 2)
                            beta_coeff -= left_prev_op * curr_feed_history[post_idx] * op_posteri
                            const_coeff -= curr_feed_history[post_idx]
                            const_coeff += op_posteri
                        if curr_feed_epoch[post_idx] > next_epoch:
                            next_epoch += 1
                    left_beta = const_coeff / beta_coeff 
                    
                    # Considering right extreme of the 95% confidence interval from the previous estimation
                    right_prev_op = prev_op + 2 * prev_sd
                    right_beta = 0.0
                    
                    next_epoch = estim_epoch.get(node_id, 0)
                    
                    # computing beta and constant coefficients in the numerator
                    # of the fraction obtained by computing
                    # ABEBA(yi(t+1)) - yi(t+1) = 0 
                    beta_coeff = right_prev_op ** 3
                    beta_coeff -= (right_prev_op ** 2) * op_posteri
                    const_coeff = -right_prev_op
                    const_coeff += op_posteri
                    for post_idx in range(len(curr_feed_epoch)):
                        if curr_feed_epoch[post_idx] >= next_epoch:
                            beta_coeff += right_prev_op * (curr_feed_history[post_idx] ** 2)
                            beta_coeff -= right_prev_op * curr_feed_history[post_idx] * op_posteri
                            const_coeff -= curr_feed_history[post_idx]
                            const_coeff += op_posteri

                        if curr_feed_epoch[post_idx] > next_epoch:
                            next_epoch += 1
                    right_beta = const_coeff / beta_coeff
                    
                    # Beta estimated as the mean of beta obtained by the left extreme
                    # of the 95% confidence interval and the right one.
                    # The standard deviation of this estimate will be the 
                    # difference between the two values, divided by 4.
                    pred_beta = (left_beta + right_beta) / 2
                    pred_var = (abs(right_beta - left_beta) / 4) ** 2
                    
                    next_epoch += 1
                    estim_epoch[node_id] = next_epoch

                    # Opinion a posteri (represents the last estimation)
                    beta_posteri = estimated_beta.get(node_id, pred_beta)
                    # Error a posteri (represents the last error value)
                    err_posteri = var_beta.get(node_id, pred_var)
                    
                    # Prediction phase
                    # Using last posteri values (adding variance to error) as priori in the new epoch
                    beta_priori = beta_posteri
                    err_priori = err_posteri + variance

                    # Correction phase
                    # measurement update
                    K = err_priori/(err_priori + R)
                    # Compute new opinion and error posteri
                    beta_posteri = beta_priori + K * (pred_beta - beta_priori)
                    err_posteri = (1 - K) * err_priori

                    # Updating beta's estimate
                    estimated_beta[node_id] = beta_posteri
                    var_beta[node_id] = err_posteri

                # Updating values obtained
                estimated[node_id] = op_posteri
                posteri_opinion[node_id] = op_posteri
                posteri_error[node_id] = P_posteri

                
            
            to_estim[node_id] = []
        # Updating estimated opinions
        nx.set_node_attributes(G, posteri_opinion, name='posteri_opinion')
        nx.set_node_attributes(G, posteri_error, name='posteri_error')
        nx.set_node_attributes(G, estimated_beta, name='estimated_beta')
        nx.set_node_attributes(G, var_beta, name='var_beta')
        nx.set_node_attributes(G, estim_epoch, name='estim_epoch')
    nx.set_node_attributes(G, to_estim, name='to_estimate')  
    nx.set_node_attributes(G, estimated, name='estimated_opinion')
    return G

