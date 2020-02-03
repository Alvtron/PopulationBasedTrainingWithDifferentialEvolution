import random

import context
from pbt.hyperparameters import Hyperparameter, Hyperparameters
from pbt.utils.iterable import unwrap_iterable

def hyperparameter_math_test():
    print("Hyperparameter Math Test")
    a = Hyperparameter(0, 100)
    b = Hyperparameter(0, 100)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"{a} + {b} = {a + b}")
    print(f"{a} - {b} = {a - b}")
    print(f"{a} * {b} = {a * b}")
    print(f"{a} / {b} = {a / b}")
    print(f"{a} ** {b} = {a ** b}")
    print("--")
    print(f"{a} + 0.1 = {a + 0.1}")
    print(f"{a} - 0.1 = {a - 0.1}")
    print(f"{a} * 2 = {a * 2}")
    print(f"{a} / 2 = {a / 2}")
    print(f"{a} ** 2 = {a ** 2}")
    print("--")
    a.normalized = 0.5
    print(f"a.normalized = 0.5 --> a.normalized: {a.normalized}, a.value: {a.value}")
    a.value = 75
    print(f"a.value = 75 --> a.normalized: {a.normalized}, a.value: {a.value}")
    a.value = 130
    print(f"a.value = 130 --> a.normalized: {a.normalized}, a.value: {a.value}")
    a.value = -0.40
    print(f"a.value = -0.40 --> a.normalized: {a.normalized}, a.value: {a.value}")

def categorical_hyperparameter_test():
    print("Categorical Hyperparameter Test")
    categorical = Hyperparameter("A","B","C","D","E","F","G","H","I", is_categorical = True)
    print("sampling uniform...")
    for _ in range(3):
        categorical.sample_uniform()
        print(f"{categorical.normalized}: {categorical.value}")
    print("perturbing...")
    for _ in range(3):
        random_value = random.choice((0.8, 1.2))
        categorical *= random_value
        print(f"{categorical.normalized}: {categorical.value}")
    print("--")
    print(f"Normalized: {categorical.normalized}")
    print(f"Value: {categorical.value}")
    print(f"Lower bound: {categorical.lower_bound}")
    print(f"Upper bound: {categorical.upper_bound}")
    print("--")
    print(f"Normalized value {0.30} gives me value {categorical.from_normalized(0.30)}")
    for value in categorical.search_space:
        print(f"Value {value} gives me normalized value {categorical.from_value(value)}")
    print("--")
    print("Tuple Categorical Hyperparameter Test")
    # tuples in categorical hyper-parameter
    tuple_hp = Hyperparameter((0.9, 0.999), is_categorical=True)
    
    print("--")
    print(f"Normalized: {tuple_hp.normalized}")
    print(f"Value: {tuple_hp.value}")
    print(f"Lower bound: {tuple_hp.lower_bound}")
    print(f"Upper bound: {tuple_hp.upper_bound}")
    print("--")

def hyperparameter_configuration_test():
    print("Hyperparameter Configuration Test")
    hp1 = Hyperparameters(
            augment_params = {
                'a': Hyperparameter(1, 256)},
            model_params = {
                'b': Hyperparameter(1e-6, 1e-0)},
            optimizer_params = {
                'c': Hyperparameter(0.0, 1e-5),
                'd': Hyperparameter(False, True, is_categorical = True)
                })
    hp2 = Hyperparameters(
            augment_params = {
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

    print(hp1.values())
    print(hp1.normalized)

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

    print("hp1 - 0.5:")
    c = hp1 - 0.5

    for param_name, param_value in c:
        print(param_name, param_value)

    print("hp1 += hp2:")
    hp1 += hp2

    for param_name, param_value in hp1:
        print(param_name, param_value)

    print("--")
    print("Hyperparameter Alteration Test")
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
    print("optimizer value dict:")
    print(hp1.get_optimizer_value_dict())

    print("--")
    print("print with for loop with indices:")
    for index in range(len(hp1)):
        print(f"hp1[{index}]: {hp1[index]}")
    print("change with for loop with indices:")
    for index in range(len(hp1)):
        hp1[index] = Hyperparameter(1, 256)
        print(f"hp1[{index}]: {hp1[index]}")

if __name__ == "__main__":
    hyperparameter_math_test()
    print("____________________")
    categorical_hyperparameter_test()
    print("____________________")
    hyperparameter_configuration_test()