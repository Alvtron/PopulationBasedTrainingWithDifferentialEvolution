
# Storn, R., & Price, K. (1995). Differential Evolution - A simple and efficient adaptive scheme for global optimization over continuous spaces. 
# Price, K., Storn, R. M., & Lampinen, J. A. (2005). Differential Evolution: A Practical Approach to Global Optimization (Natural Computing Series). Springer-Verlag.

def de_rand_1(F, x_r0, x_r1, x_r2):
    return x_r0 + F * (x_r1 - x_r2)

def de_current_to_best_1(F, x_base, x_best, x_r1, x_r2):
    return x_base + F * (x_best - x_base) + F * (x_r1 - x_r2)

def de_best_1(F, x_best, x_r1, x_r2):
    return x_best + F * (x_r1 - x_r2)

def de_best_2(F, x_best, x_r1, x_r2, x_r3, x_r4):
    return x_best + F * (x_r1 - x_r2 + x_r3 - x_r4)