import math
import random
from typing import Sequence

def randn(mean: float, std: float):
    """Generate a random value from a normal distribution."""
    return random.normalvariate(mean, std)

def randc(mean: float, std: float):
    """Generate a random value from a Cauchy distribution."""
    p = 0.0
    while p == 0.0:
        p = random.random()
    return mean + std * math.tan(math.pi * (p - 0.5))

def mean_wl(S, weights: Sequence[float]) -> float:
    """
    The weighted Lehmer mean of a tuple x of positive real numbers,
    with respect to a tuple w of positive weights.
    """
    def weight(weights, k):
        return weights[k] / sum(weights)
    A = sum(weight(weights, k) * s**2 for k, s in enumerate(S))
    B = sum(weight(weights, k) * s for k, s in enumerate(S))
    return A / B