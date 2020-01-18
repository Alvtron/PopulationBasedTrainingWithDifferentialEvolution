import math
import random

def randn(mean, std):
    """Generate a random value from a normal distribution."""
    return random.normalvariate(mean, std)

def randc(mean, std):
    """Generate a random value from a Cauchy distribution."""
    p = 0.0
    while p == 0.0:
        p = random.random()
    return mean + std * math.tan(math.pi * (p - 0.5))