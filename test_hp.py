import random
import itertools
from hyperparameters import Hyperparameter, Hyperparameters
from utils import unwrap_iterable

def hyperparameter_math_testing():
    a = Hyperparameter(0, 1000)
    b = Hyperparameter(0, 1000)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * b = {a * b}")
    print(f"a / b = {a / b}")
    print(f"a ** b = {a ** b}")

    print(f"a + 0.1 = {a + 0.1}")
    print(f"a - 0.1 = {a - 0.1}")
    print(f"a * 2 = {a * 2}")
    print(f"a / 2 = {a / 2}")
    print(f"a ** 2 = {a ** 2}")

def categorical_hyperparameter_testing():
    categorical = Hyperparameter("A","B","C","D","E","F","G","H","I", is_categorical = True)

    print("sampling uniform...")
    for i in range(3):
        categorical.sample_uniform()
        print(f"{categorical.normalized()}: {categorical.value()}")

    print("perturbing...")
    for i in range(3):
        random_value = random.choice((0.8, 1.2))
        categorical *= random_value
        print(f"{categorical.normalized()}: {categorical.value()}")

    print("--")
    print(f"Normalized: {categorical.normalized()}")
    print(f"Value: {categorical.value()}")
    print(f"Lower bound: {categorical.get_lower_bound()}")
    print(f"Upper bound: {categorical.get_upper_bound()}")
    print("--")
    print(f"Normalized value {0.30} gives me value {categorical.get_value(0.30)}")
    for value in categorical.search_space:
        print(f"Value {value} gives me normalized value {categorical.get_normalized_value(value)}")

def hyperparameter_configuration_testing():
    hp1 = Hyperparameters(
            general_params = {
                'a': Hyperparameter(1, 256)},
            model_params = {
                'b': Hyperparameter(1e-6, 1e-0)},
            optimizer_params = {
                'c': Hyperparameter(0.0, 1e-5),
                'd': Hyperparameter(False, True, is_categorical = True)
                })
    hp2 = Hyperparameters(
            general_params = {
                'a': Hyperparameter(1, 256)},
            model_params = {
                'b': Hyperparameter(1e-6, 1e-0)},
            optimizer_params = {
                'c': Hyperparameter(0.0, 1e-5),
                'd': Hyperparameter(False, True, is_categorical = True)
                })

    print(len(hp1))
    print(hp1 == hp2)
    print(hp1 != hp2)

    for param_name, param_value in hp1:
        print(param_name, param_value)
    for param_name, param_value in hp2:
        print(param_name, param_value)

    print("hp1 + hp2:")
    a = hp1 + hp2

    for param_name, param_value in a:
        print(param_name, param_value)

    print("hp1 + 0.5:")
    b = hp1 + 0.5

    for param_name, param_value in b:
        print(param_name, param_value)

    print("hp1 += hp2:")
    hp1 += hp2

    for param_name, param_value in hp1:
        print(param_name, param_value)

    print("--")

    params = [
        Hyperparameter(1, 256),
        Hyperparameter(1e-6, 1e-0),
        Hyperparameter(0.0, 1e-5),
        Hyperparameter(False, True, is_categorical = True)]

    hp1.set(params)
    for param_name, param_value in hp1:
        print(param_name, param_value)
    
    print("--")

    print("hp1[1] =", hp1[1])
    print("changing hp1[1]...")
    hp1[1] = Hyperparameter(1e-6, 1e-0)
    print("hp1[1] =", hp1[1])

    print("--")
    print(hp1.get_optimizer_value_dict())

if __name__ == "__main__":
    hyperparameter_math_testing()
    categorical_hyperparameter_testing()
    hyperparameter_configuration_testing()