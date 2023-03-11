import json
import numpy as np
import os
import pandas as pd
import time
import warnings
import xlsxwriter

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional ,Union


def load_json(path: str) -> Union[Dict[Any, Any], List[Any]]:
    """
    Load a json file as a dictionary or a list.

    :param path: path to the file
    :return: the object
    """
    with open(path, "r") as f:
        return json.load(f)


@contextmanager
def track_time(msg: Optional[str] = None) -> Generator[None, None, None]:
    """
    Tracks the time passed within the context manager.

    :param msg: extra message to write in the output
    :return: the context manager
    """

    print(f"Starting {msg}...")
    start = time.monotonic()

    yield  

    print(f"Ended {msg} in {time.monotonic() - start:.2f} seconds")



def _get_epoch_metrics(runs, epoch, label):
    """
    Get metrics for all runs for a given epoch.
    
    :param runs: all the runs of a configuration
    :param epoch: the index of the epoch considered
    :param label: label used for epoch identification
    :return: dataframe containing epoch metrics
    """
    
    metric_names = (
        "echo_chamber_value",  
    )

    metrics = {
        metric_name: 
            tuple(
                run["epochs_data"][epoch]["metrics"][metric_name] 
                for run in runs
            )
        for metric_name in metric_names
    }
    
    epoch_metrics = {"label": label}


    if "inf" in tuple(metric["mean"] for metric in metrics["echo_chamber_value"]):
        epoch_metrics["echo_chamber_mean"] = "inf" 
        epoch_metrics["echo_chamber_var"] = "inf"
    else:
        epoch_metrics["echo_chamber_mean"] = np.mean(tuple(metric["mean"] for metric in metrics["echo_chamber_value"])) 
        epoch_metrics["echo_chamber_var"] = (
            np.sum(np.var(tuple(metric["single_values"].values())) for metric in metrics["echo_chamber_value"]) 
            / (len(metrics["echo_chamber_value"]) ** 2)
        )

    return epoch_metrics



def _retrieve_metrics(initial_model, runs):
    """
    Retrieve initial, middle and final metrics for the given runs

    :param initial_model: model data
    :param runs: data containing runs data
    :return: dataframe containing metrics for the runs
    """
    metrics_results = list()
    for epoch in range(len(runs[0]["epochs_data"])):
        metrics = _get_epoch_metrics(runs=runs, epoch=epoch, label=f"Epoch {epoch}")

        metrics_results.append(metrics)
    
    dataframe = pd.DataFrame.from_records(tuple(metrics_results))


    dataframe["estim_strategy"] = initial_model["params"]["estim_strategy"]
    dataframe["content_strategy"] = initial_model["params"]["strategy_content_recommender"]
    dataframe["people_strategy"] = initial_model["params"]["strategy_people_recommender"]
    

    return dataframe

def _retrieve_configuration_metrics(configuration_path):
    """
    Retrieve the metrics from the given configuration.
    
    :param configuration_path: the path to the configuration output
    :return: dataframe containing the metrics retrieved
    """
    INITIAL_MODEL = "initial_model_configuration.json"
    RUNS = 50


    run_paths = tuple(f"model_configuration_result{i}.json" for i in range(RUNS))    
    
    with track_time(msg="importing data from"):
        runs = tuple(load_json(path= configuration_path + run_path) for run_path in run_paths)
        initial_model = load_json(path=configuration_path + INITIAL_MODEL)
        
    with track_time(msg="retrieving metrics"):
        dataframe = _retrieve_metrics(
            initial_model=initial_model,
            runs=runs,
        )

    return dataframe

def main() -> None:
    # Add your base_path to the outputs
    BASE_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/random_comparison_2/"
    OUTPUT_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/random_comparison_output_2.xlsx"

    warnings.filterwarnings("ignore")

    df = pd.DataFrame()

    configurations = tuple(os.listdir(BASE_PATH))

    for configuration in configurations:
        df = df.append(_retrieve_configuration_metrics(configuration_path= BASE_PATH + configuration + "/"))
        
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        df.to_excel(writer) 

    

if __name__ == "__main__":
    main()
