import random

def random_reinitialization(mutant, lower_bounds, upper_bounds):
    """
    Page 204, Price, K., Storn, R. M., & Lampinen, J. A. (2005).
    Differential Evolution: A Practical Approach to Global Optimization (Natural Computing Series). Springer-Verlag.
    """
    if mutant < lower_bounds or mutant > upper_bounds:
        mutant = lower_bounds + random.uniform(0.0, 1.0) * (upper_bounds - lower_bounds)
    return mutant

def bounce_back(base, mutant, lower_bounds, upper_bounds):
    """
    Page 205, Price, K., Storn, R. M., & Lampinen, J. A. (2005).
    Differential Evolution: A Practical Approach to Global Optimization (Natural Computing Series). Springer-Verlag.
    """
    if mutant < lower_bounds:
        return base + random.uniform(0.0, 1.0) * (lower_bounds - base)
    if mutant > upper_bounds:
        return base + random.uniform(0.0, 1.0) * (upper_bounds - base)
    return mutant

def halving(base, mutant, lower_bounds, upper_bounds):
    """
    Constrain method used in used in the original SHADE paper.\n 
    Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for Differential Evolution.\n
    2013 IEEE Congress on Evolutionary Computation, 71–78. https://doi.org/10.1109/CEC.2013.6557555
    """
    if mutant < lower_bounds:
        return (lower_bounds + base) / 2
    if mutant > upper_bounds:
        return (upper_bounds + base) / 2
    return mutant