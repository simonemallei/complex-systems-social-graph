import json
import time

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
