import math
import random
from bisect import bisect
from itertools import accumulate
from typing import List


def chunks(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def choice_with_weight(population, weights):
    n = len(population)
    try:
        cum_weights = list(accumulate(weights))
    except TypeError:
        if not isinstance(weights, int):
            raise
        k = weights
        raise TypeError(
            f"The number of choices must be a keyword argument: {k=}"
        ) from None
    if len(cum_weights) != n:
        raise ValueError("The number of weights does not match the population")
    total = cum_weights[-1] + 0.0  # convert to float
    if total <= 0.0:
        raise ValueError("Total of weights must be greater than zero")
    if not math.isfinite(total):
        raise ValueError("Total of weights must be finite")
    hi = n - 1
    while True:
        yield population[bisect(cum_weights, random.random() * total, 0, hi)]
