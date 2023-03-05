import json
import time
from contextlib import contextmanager

import numpy
from scipy import special
from typing import List, Union, Dict, Any, Optional, Generator


def softmax_with_temperature(
    logits: List[float], temperature: float = 0.05
) -> numpy.ndarray:
    logits_array = numpy.array(logits)
    logits_temp = logits_array / temperature

    distribution = special.softmax(logits_temp)
    return distribution


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
