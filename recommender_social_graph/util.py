import numpy
from scipy import special
from typing import List


def softmax_with_temperature(logits: List[float], temperature: float = 0.05) -> numpy.ndarray:
    logits_array = numpy.array(logits)
    logits_temp = logits_array / temperature

    distribution = special.softmax(logits_temp)
    return distribution