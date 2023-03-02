from utilities import load_json, track_time
from run_simulator import RunSimulator

def _simulate_runs(initial_graph, runs):
    """
    Simulate the given runs from the given initial graph.

    :param initial_graph: initial graph data
    :param runs: runs data
    """
    for run in runs:
        simulator = RunSimulator(
            initial_graph=initial_graph,
            run=run,
        )
        graphs = simulator.simulate_epochs()
        for graph in graphs:
            #TODO: Use the graphs obtained to obtain the missing/wrong metrics (for example echo chamber)
            #TODO: We can update simulate_epochs and make it return the graph for each epoch with the evaluated metrics data!
            i = 2 # This is a placeholder line in order to be sure that each epoch is simulated
            

def main() -> None:
    # Add your base_path to the outputs
    BASE_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/output/"
    CONFIGURATION = "2023-02-01_12-12-19_config_0/"
    INITIAL_GRAPH = "initial_graph_configuration.json"
    RUNS = 50

    run_paths = tuple(f"model_configuration_result{i}.json" for i in range(RUNS))    
    
    with track_time(msg="importing data"):
        runs = tuple(load_json(path=BASE_PATH + CONFIGURATION + run_path) for run_path in run_paths)
        initial_graph = load_json(path=BASE_PATH + CONFIGURATION + INITIAL_GRAPH)
    
    with track_time(msg="simulating all runs"):
        _simulate_runs(initial_graph=initial_graph, runs=runs)


if __name__ == "__main__":
    main()
