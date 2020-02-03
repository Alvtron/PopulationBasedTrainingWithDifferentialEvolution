import random

def random_reinitialization(mutant, lower_bounds, upper_bounds):
    """Page 204, DE(2006)"""
    if mutant < lower_bounds or mutant > upper_bounds:
        mutant = lower_bounds + random.uniform(0.0, 1.0) * (upper_bounds - lower_bounds)
    return mutant

def bounce_back(base, mutant, lower_bounds, upper_bounds):
    """Page 205, DE(2006)"""
    if mutant < lower_bounds:
        return base + random.uniform(0.0, 1.0) * (lower_bounds - base)
    if mutant > upper_bounds:
        return base + random.uniform(0.0, 1.0) * (upper_bounds - base)
    return mutant

def halving(base, mutant, lower_bounds, upper_bounds):
    """Page ???, DE(2006), used in SHADE"""
    if mutant < lower_bounds:
        return (lower_bounds + base) / 2
    if mutant > upper_bounds:
        return (upper_bounds + base) / 2
    return mutant