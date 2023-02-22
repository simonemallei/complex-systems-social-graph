INPUT EXPLANATION
If you're using Visual Studio Code, place the launch.json file in the .vscode folder. It will have to be created, if not present, in the complex-systems-social-graph folder.
The main file from which the execution of the program starts is parallel_simulation.py. It receives as input a path where the program expects to find the "configurations" folder. Inside this folder the files graph-configurations.json and model-configurations.json must be placed. The first of the two files contains the configurations of the graphs to be created, the second contains the configurations of the model that will be used in the simulations.

graph-configurations.json explanation:
It is a json file formed by an array of objects having the same keys. Each object contains the configuration of a different graph.
Below is the explanation of each item key:
- n_nodes :  Number of nodes in the graph
- beta : array containing the beta parameter values ​​of the beba/abeba model. Range: (0,1]
- avg_friend : average number of friends for each node of the graph
- prob_post : array containing the base probability that each node can publish. Range: [0,1]
- hp_alpha : homophily parameter
- hp_beta : preferential attachment parameter

model-configurations.json explanation:
It is a json file formed by an array of objects having the same keys. Each object contains a configuration of the abeba model.
Below is the explanation of each item key:
- n_runs : number of executions that will be launched using the parameters contained in the "params" key
- params : object that contains a parameter configuration of the abeba model.
  Below is the explanation of the keys of this object:
    - graph_id : id of the graph that you want to use with this configuration. It is practically the index of the chosen   graph configuration. Note that it starts at 0.
    - num_epochs : duration in epochs of the elaboration of this configuration 
    - rate_updating_nodes : probability that a node will update its opinion. Range: [0,1]
    - stubborness : stubborness coefficient. How much the nodes remain close to their previous opinion. In this way, the nodes converge slower than before,     giving the recommenders the chance to influence the graph evolution. Range: [0,1]. Default: 0.75
    - post_epsilon: The Gaussian noise's standard deviation. This is the noise that is applied to the opinion when a node publishes content
    - estim_strategy: strategy used to estimate the opinion of a node. Values accepted: "base", "kalman". Default: "base"
    - strat_param:
        The dictionary containing the parameters value used by the recommender, based
        on {strategy} value.
            - key: "alpha",
              value: alpha coefficient used in "base" strategy. Default: 0.9
            - key: "variance",
              value: process variance of "kalman" strategy. Default: 1e-5
            - key: "variance_measure",
              value: measure variance of "kalman" strategy. Default: 0.1 ** 2 (0.1 ^ 2)
    - strategy_content_recommender: The string that defines the strategy used by the content recommender. 
    Values accepted: "no_recommender", "random", "normal", "nudge", "similar", "unsimilar". Default: "random"
    - strat_param_content_recommender: The dictionary containing the parameters value used by the recommender, based
    on {strategy} value.
        - key: "n_post",
            value: number of posts added in activated node's feed by the recommender.
        - key: "normal_mean",
            value: mean of distribution of values produced by "normal" strategy.
        - key: "normal_std",
            value: standard dev. of distribution of values produced by "normal" strategy.
        - key: "nudge_goal",
            value: the "opinion goal" by the nudging content recommender.
        - key: "similar_thresh",
            value: threshold value used by "similar" strategy.
        - key: "unsimilar_thresh",
            value: threshold value used by "unsimilar" strategy.
    - strategy_people_recommender:  The string that defines the strategy used by the people recommender. Values accepted: "random", "opinion_estimation_based", "topology_based", "opinion_estimation_topology_mixed". Default "random"
    - substrategy_people_recommender: The string that defines the sub-strategy applied to the chosen strategy, excluding the random strategy. Values accepted: "counteract_homophily", "favour_homophily". Default: "None", because the default strategy is the random one.
    - strat_param_people_recommender: {dictionary} default: {"connected_components": true}
        dictionary that containing the parameters value used by the recommender. In the current version, the only strategy using this dictionary is opinion_estimation_topology_mixed. 
        Elements:
        Connected_components: {false or true}
        If the value is True, then the opinion_estimation_topology_mixed strategy will connect the components of the graph before choosing who to recommend (using both main strategies), as is the case for the topology_based strategy.
        If the value is False, the opinion_estimation_topology_mixed strategy will always use both main strategies, but the contribution of the topology_based strategy will be limited only to the nodes present in the considered component.

OUTPUT EXPLANATION
The program saves the results in the "output" folder which, if it doesn't exist, will be created inside the path given in input when the program is started.
Within this folder, as many sub-folders will be created as there are configurations specified in the model-configurations.json file.
In each sub-folder there will be a number of output files equal to the number of runs for that specific configuration. This number is always set in the model-configurations.json file.
Note that if an exception were to occur within the program, all the output files of that processing would be deleted, so as not to have partial results of the processing.