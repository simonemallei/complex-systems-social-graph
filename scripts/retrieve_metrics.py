import json
import numpy as np
import os
import pandas as pd
import time
import warnings
import xlsxwriter

from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional ,Union

_EPOCHS = (0, 19, 39, 59, 79, 99)

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
        "polarization_value", 
        "bimodality", 
        "disagreement",
        "echo_chamber_value",  
        "feed_entropy", 
        "feed_satisfaction",
        "recommendation_homophily_rate",
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

    epoch_metrics["polarization_mean"] = np.mean(metrics["polarization_value"]) 
    epoch_metrics["polarization_var"] = np.std(metrics["polarization_value"])

    epoch_metrics["bimodality_mean"] = np.mean(metrics["bimodality"]) 
    epoch_metrics["bimodality_var"] = np.var(metrics["bimodality"])

    epoch_metrics["disagreement_mean"] = np.mean(tuple(metric["mean"] for metric in metrics["disagreement"])) 
    epoch_metrics["disagreement_var"] = (
        np.sum(tuple(metric["variance"] for metric in metrics["disagreement"]))
        / (len(metrics["disagreement"]) ** 2)
    )

    if "inf" in tuple(metric["mean"] for metric in metrics["echo_chamber_value"]):
        epoch_metrics["echo_chamber_mean"] = "inf" 
        epoch_metrics["echo_chamber_var"] = "inf"
    else:
        epoch_metrics["echo_chamber_mean"] = np.mean(tuple(metric["mean"] for metric in metrics["echo_chamber_value"])) 
        epoch_metrics["echo_chamber_var"] = (
            np.sum(np.var(tuple(metric["single_values"].values())) for metric in metrics["echo_chamber_value"]) 
            / (len(metrics["echo_chamber_value"]) ** 2)
        )

    filtered_length = len(tuple(metric["variance"] for metric in metrics["feed_entropy"] if metric["variance"] is not None))
    epoch_metrics["feed_entropy_mean"] = np.mean(tuple(metric["mean"] for metric in metrics["feed_entropy"] if metric["mean"] is not None)) 
    epoch_metrics["feed_entropy_var"] = (
        np.sum(tuple(metric["variance"] for metric in metrics["feed_entropy"] if metric["variance"] is not None)) 
        / (filtered_length ** 2)
    )

    epoch_metrics["feed_satisfaction_mean"] = np.mean(tuple(np.mean(tuple(metric.values())) for metric in metrics["feed_satisfaction"])) 
    epoch_metrics["feed_satisfaction_var"] = (
        np.sum(np.var(tuple(metric.values())) for metric in metrics["feed_satisfaction"]) 
        / (len(metrics["feed_satisfaction"]) ** 2)
    )

    filtered_length = len(tuple(metric["variance"] for metric in metrics["recommendation_homophily_rate"] if metric["variance"] is not None))
    epoch_metrics["recommendation_homophily_rate_mean"] = np.mean(
        tuple(metric["mean"] for metric in metrics["recommendation_homophily_rate"] if metric["mean"] is not None)
    )
    epoch_metrics["recommendation_homophily_rate_var"] = (
        np.sum(tuple(metric["variance"] for metric in metrics["recommendation_homophily_rate"] if metric["variance"] is not None)) 
        / (filtered_length ** 2)
    )
    return epoch_metrics



def _retrieve_metrics(initial_model, runs):
    """
    Retrieve initial, middle and final metrics for the given runs

    :param initial_model: model data
    :param runs: data containing runs data
    :return: dataframe containing metrics for the runs
    """
    metrics = tuple(
        _get_epoch_metrics(runs=runs, epoch=epoch, label=f"Epoch {epoch}")
        for epoch in _EPOCHS
    )

    dataframe = pd.DataFrame.from_records(metrics)


    dataframe["content_strategy"] = initial_model["params"]["strategy_content_recommender"]
    dataframe["people_strategy"] = initial_model["params"]["strategy_people_recommender"]
    dataframe["people_substrategy"] = initial_model["params"]["substrategy_people_recommender"]
    if len(initial_model["params"]["strat_param_people_recommender"]) == 0:
        dataframe["people_connected_components"] = None
        dataframe["configuration_name"] = (
            f"{initial_model['params']['strategy_content_recommender']}/"
            f"{initial_model['params']['strategy_people_recommender']}-"
            f"{initial_model['params']['substrategy_people_recommender']}" 
        )
    else:
        dataframe["people_connected_components"] = initial_model["params"]["strat_param_people_recommender"]["connected_components"]
        dataframe["configuration_name"] = (
            f"{initial_model['params']['strategy_content_recommender']}/"
            f"{initial_model['params']['strategy_people_recommender']}-"
            f"{initial_model['params']['substrategy_people_recommender']}-"
            f"{initial_model['params']['strat_param_people_recommender']['connected_components']}" 
        )
    
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

def write_excel(sheet, row, col, value):
    if type(value) != str and np.isnan(value):
        sheet.write(row, col, "NaN")
    elif type(value) != str and np.isinf(value):
        sheet.write(row, col, "Inf")
    else:
        sheet.write(row, col, value)


def write_metrics(output_path, df):
    workbook = xlsxwriter.Workbook(output_path)

    METRICS = (
        "polarization",
        "bimodality",
        "disagreement",
        "echo_chamber",
        "feed_entropy",
        "feed_satisfaction",
        "recommendation_homophily_rate",
    )

    CONFIGURATION_PARAMETERS = (
        "configuration_name",
        "content_strategy",
        "people_strategy",
        "people_substrategy",
        "people_connected_components",
    )

    for metric in METRICS:
        ATTRIBUTES = (
            f"{metric}_mean",
            f"{metric}_var",
        )



        curr_sheet = workbook.add_worksheet(metric)
        base = 1
        for idx, param in enumerate(CONFIGURATION_PARAMETERS):
            curr_sheet.write(0, idx, param)
            for row in range(0, len(df.index), len(_EPOCHS)):
                curr_sheet.write(base + (row // len(_EPOCHS)), idx, df[param].iloc[row])

        offset = len(CONFIGURATION_PARAMETERS)
        for idx, attribute in enumerate(ATTRIBUTES):
            for epoch_idx, epoch in enumerate(_EPOCHS):
                curr_sheet.write(
                    0, 
                    offset + idx * len(_EPOCHS) + epoch_idx, 
                    f"{attribute}_epoch_{epoch}"
                )
                for row in range(0, len(df.index), len(_EPOCHS)):
                    write_excel(
                        curr_sheet, 
                        base + (row // len(_EPOCHS)), 
                        offset + idx * len(_EPOCHS) + epoch_idx, 
                        df[attribute].iloc[row+epoch_idx]
                    )

    workbook.close()
    

def main() -> None:
    # Add your base_path to the outputs
    BASE_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/output/"
    OUTPUT_PATH = "D:/Projects/test_complex_system/complex-systems-social-graph/recommender_social_graph/metrics_output_5_values.xlsx"

    warnings.filterwarnings("ignore")

    df = pd.DataFrame()

    configurations = tuple(os.listdir(BASE_PATH))

    for configuration in configurations:
        df = df.append(_retrieve_configuration_metrics(configuration_path= BASE_PATH + configuration + "/"))
        
    write_metrics(output_path=OUTPUT_PATH, df=df)
    

if __name__ == "__main__":
    main()
