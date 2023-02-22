from enum import Enum
import itertools
import json


class PeopleStrategy (str, Enum):
    NO_RECOMMENDER = "no_recommender"
    RANDOM = "random"
    OPINION_ESTIMATION = "opinion_estimation_based"
    TOPOLOGY = "topology_based"
    OPINION_TOPOLOGY = "opinion_estimation_topology_mixed"


class PeopleSubStrategy(str, Enum):
    COUNTERACT_HOMOPHILY = "counteract_homophily"
    FAVOUR_HOMOPHILY = "favour_homophily"


class ContentStrategy (str, Enum):
    NO_RECOMMENDER = "no_recommender"
    NORMAL ="normal"
    NUDGE = "nudge"
    RANDOM = "random"
    SIMILAR =  "similar"
    UNSIMILAR = "unsimilar"


class Params(str, Enum):
    RATE_UPDATING_NODES = "rate_updating_nodes"
    POST_EPSILON = "post_epsilon"
    STUBBORNESS = "stubborness"
    ESTIM_STRATEGY = "estim_strategy"
    ESTIM_STRATEGY_PARAM = "estim_strategy_param"
    STRAT_CONTENT = "strategy_content_recommender"
    STRAT_CONTENT_PARAM = "strat_param_content_recommender"
    STRAT_PEOPLE = "strategy_people_recommender"
    SUBSTRAT_PEOPLE = "substrategy_people_recommender"
    STRAT_PEOPLE_PARAM = "strat_param_people_recommender"


def _get_people_config():
    """
    Return a tuple with all the possible configuration for people strategies.

    :return: people strategies configuration
    """
    _SUB_COMPATIBLE = (
        PeopleStrategy.OPINION_ESTIMATION,
        PeopleStrategy.TOPOLOGY,
    )
    _CONNECTED_COMPONENTS = (False, True)

    # Opinion Topology Mixed is the only strategy that needs all 
    # the parameters config
    complete_params = tuple(
        {
            Params.STRAT_PEOPLE: PeopleStrategy.OPINION_TOPOLOGY,
            Params.SUBSTRAT_PEOPLE: substrategy,
            Params.STRAT_PEOPLE_PARAM: {
                "connected_components": flag,
            }
        }
        for substrategy in PeopleSubStrategy
        for flag in _CONNECTED_COMPONENTS    
    )
    # The strategies that need the substrategy as well
    sub_compatible = tuple(
        {
            Params.STRAT_PEOPLE: strategy, 
            Params.SUBSTRAT_PEOPLE: substrategy,
            Params.STRAT_PEOPLE_PARAM: dict(),
        }
        for strategy in PeopleStrategy
        for substrategy in PeopleSubStrategy
        if strategy in _SUB_COMPATIBLE
    )
    # All the remaining strategies
    sub_incompatible = tuple(
        {
            Params.STRAT_PEOPLE: strategy,
            Params.SUBSTRAT_PEOPLE: "",
            Params.STRAT_PEOPLE_PARAM: dict(),
        }
        for strategy in PeopleStrategy
        if strategy not in _SUB_COMPATIBLE + (PeopleStrategy.OPINION_TOPOLOGY,)
    )
    config = complete_params + sub_compatible + sub_incompatible

    return config


def _get_content_config():
    """
    Return a tuple with all the possible configuration for content strategies.

    :return: content strategies configuration
    """
    _N_POST = 1
    _NORMAL_STD = 0.1
    _MEANS = (0.0,)
    _GOALS = (0.0,)
    _SIMILAR_THRESHOLD = 0.4
    _UNSIMILAR_THRESHOLD = 0.4

    # For specific tests
    #_MEANS = (-0.6, 0.0, 0.6)
    #_GOALS = (-0.6, 0.0, 0.6)
    

    strategy_params = {
        ContentStrategy.NO_RECOMMENDER: ({},),
        ContentStrategy.RANDOM: ({"n_post": _N_POST},),
        ContentStrategy.NORMAL: tuple(
            {"n_post": _N_POST, "normal_std": _NORMAL_STD, "normal_mean": mean}
            for mean in _MEANS
        ),
        ContentStrategy.NUDGE: tuple(
            {"n_post": _N_POST, "nudge_goal": goal}
            for goal in _GOALS
        ),
        ContentStrategy.SIMILAR: ({"similar_thresh": _SIMILAR_THRESHOLD},),
        ContentStrategy.UNSIMILAR: ({"unsimilar_thresh": _UNSIMILAR_THRESHOLD},)
    }
    config = tuple(
        {
            Params.STRAT_CONTENT: strategy, 
            Params.STRAT_CONTENT_PARAM: strategy_param,
        } 
        for strategy in ContentStrategy
        for strategy_param in strategy_params[strategy]
    )
    return config

    
def _get_simulation_config():
    """
    Return a tuple with all the possible configuration for the execution of simulation.

    :return: simulation configuration
    """
    _RATE_UPDATING_NODES = (0.5,)
    _POST_EPSILON = (0.1,)
    _BASE_ALPHA = ()
    _STUBBORNESS = (0.75,)
    _VARIANCE = 1e-5
    _VARIANCE_MEASURE = 0.1 ** 2

    # For specific tests
    #_BASE_ALPHA = (0.1, 0.5, 0.9)

    _ESTIM_STRATEGY = tuple(
        {
            Params.ESTIM_STRATEGY: "base",
            Params.ESTIM_STRATEGY_PARAM: {
                "alpha": alpha,
            },
        }
        for alpha in _BASE_ALPHA 
    ) + ({
        Params.ESTIM_STRATEGY: "kalman",
        Params.ESTIM_STRATEGY_PARAM: {        
            "variance": _VARIANCE,
            "variance_measure": _VARIANCE_MEASURE,
        },
    },)

    config = tuple({
            Params.RATE_UPDATING_NODES: rate_updating_nodes,
            Params.POST_EPSILON: post_epsilon,
            Params.STUBBORNESS: stubborness,
            Params.ESTIM_STRATEGY: estim[Params.ESTIM_STRATEGY],
            Params.ESTIM_STRATEGY_PARAM: estim[Params.ESTIM_STRATEGY_PARAM],
        }
        for rate_updating_nodes in _RATE_UPDATING_NODES
        for stubborness in _STUBBORNESS
        for post_epsilon in _POST_EPSILON
        for estim in _ESTIM_STRATEGY
    )
    return config

def _get_graph_config():
    """
    Return a tuple with all the possible configuration for the graph.

    :return: graph configuration
    """
    _N_NODES = (500,)
    _BETA = ((1,),)
    _AVG_FRIENDS = tuple(n_nodes // 10 for n_nodes in _N_NODES)
    _PROB_POST = ((0.5,),)
    _HP_ALPHA_BETA = (
        {
            "hp_alpha": 2,
            "hp_beta": 1,
        },
    )

    # For specific tests
    # _HP_ALPHA_BETA = (
    #     {
    #         "hp_alpha": 2,
    #         "hp_beta": 0,
    #     },
    #     {
    #         "hp_alpha": 0,
    #         "hp_beta": 2,
    #     },
    #     {
    #         "hp_alpha": 2,
    #         "hp_beta": 2,
    #     },
    # )

    config = tuple(
        {
            "n_nodes": n_nodes,
            "beta": beta,
            "avg_friend": avg_friends,
            "prob_post": prob_post,
            "hp_alpha": hp["hp_alpha"],
            "hp_beta": hp["hp_beta"],
        }
        for (n_nodes, avg_friends) in zip(_N_NODES, _AVG_FRIENDS)
        for beta in _BETA
        for prob_post in _PROB_POST
        for hp in _HP_ALPHA_BETA
    )
    return config



def _get_model_config(index, n_runs, num_epochs, config):
    """
    Return a tuple with all the possible configuration for the model.
    
    :param index: index of the model
    :param n_runs: number of runs
    :param n_epochs: number of epochs for each run
    :param config: base config
    :return: model configuration
    """
    _GRAPH_ID = 0

    model_params =  {
        "graph_id": _GRAPH_ID,
        "num_epochs": num_epochs,
        Params.RATE_UPDATING_NODES: config["simulation_config"][Params.RATE_UPDATING_NODES],
        Params.POST_EPSILON: config["simulation_config"][Params.POST_EPSILON],
        Params.STUBBORNESS: config["simulation_config"][Params.STUBBORNESS],
        Params.ESTIM_STRATEGY: config["simulation_config"][Params.ESTIM_STRATEGY],
        Params.ESTIM_STRATEGY_PARAM: config["simulation_config"][Params.ESTIM_STRATEGY_PARAM],
        Params.STRAT_CONTENT: config["content_config"][Params.STRAT_CONTENT],
        Params.STRAT_CONTENT_PARAM: config["content_config"][Params.STRAT_CONTENT_PARAM],
        Params.STRAT_PEOPLE: config["people_config"][Params.STRAT_PEOPLE],
    } 
    model_params.update(
        (
            {Params.SUBSTRAT_PEOPLE: config["people_config"][Params.SUBSTRAT_PEOPLE]}
            if Params.SUBSTRAT_PEOPLE in config["people_config"]
            else dict()
        )
    )
    model_params.update(
        (
            {Params.STRAT_PEOPLE_PARAM: config["people_config"][Params.STRAT_PEOPLE_PARAM],}
            if Params.STRAT_PEOPLE_PARAM in config["people_config"]
            else dict()
        )
    )
    return {
        "n_runs": n_runs,
        "params": model_params,
    }


def main():
    _N_RUNS = 50
    _NUM_EPOCHS = 100

    # Get specific config
    graph_config = _get_graph_config()
    simulation_config = _get_simulation_config()
    people_config = _get_people_config()
    content_config = _get_content_config()

    parameters = (
        simulation_config,
        people_config,
        content_config,
    )
    keys = (
        "simulation_config",
        "people_config",
        "content_config",
    )
    model_configurations = tuple(itertools.product(*parameters))
    print(f"Setting {len(model_configurations)} model configs and {len(graph_config)} graph configs...")
    model_config = tuple(
        _get_model_config(
            index=index, 
            n_runs = _N_RUNS,
            num_epochs=_NUM_EPOCHS,
            config=dict(zip(keys, config)),
        )
        for index, config in enumerate(model_configurations)
    )
    # Save the model_configuration
    with open('model-configurations.json', 'w') as f:
        f.write(json.dumps(model_config, indent=2))
    # Save the graph_config
    with open('graph-configurations.json', 'w') as f:
        f.write(json.dumps(graph_config, indent=2))


if __name__ == "__main__":
    main()