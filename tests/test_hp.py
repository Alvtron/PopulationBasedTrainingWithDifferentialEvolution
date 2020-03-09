import random

import context
from pbt.hyperparameters import ContiniousHyperparameter, DiscreteHyperparameter, Hyperparameters
from pbt.utils.iterable import unwrap_iterable

def hyperparameter_math_test():
    print("Hyperparameter Math Test")
    a = ContiniousHyperparameter(0, 100)
    b = ContiniousHyperparameter(0, 100)
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
    print("Discrete Hyperparameter Test")
    categorical = DiscreteHyperparameter("A","B","C","D","E","F","G","H","I")
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
    print("Tuple Categorical ContiniousHyperparameter Test")
    # tuples in categorical hyper-parameter
    tuple_hp = DiscreteHyperparameter((0.9, 0.999))
    
    print("--")
    print(f"Normalized: {tuple_hp.normalized}")
    print(f"Value: {tuple_hp.value}")
    print(f"Lower bound: {tuple_hp.lower_bound}")
    print(f"Upper bound: {tuple_hp.upper_bound}")
    print("--")

def hyperparameter_configuration_test():
    print("Hyperparameter Configuration Test")
    hp1 = Hyperparameters(
            augment = {
                'a': ContiniousHyperparameter(1, 256)},
            model = {
                'b': ContiniousHyperparameter(1e-6, 1e-0)},
            optimizer = {
                'c': ContiniousHyperparameter(0.0, 1e-5),
                'd': DiscreteHyperparameter(False, True)
                })
    hp2 = Hyperparameters(
            augment = {
                'a': ContiniousHyperparameter(1, 256)},
            model = {
                'b': ContiniousHyperparameter(1e-6, 1e-0)},
            optimizer = {
                'c': ContiniousHyperparameter(0.0, 1e-5),
                'd': DiscreteHyperparameter(False, True)
                })

    print(len(hp1))

    print("items")
    print(list(hp1.items()))
    print("hps")
    print(list(hp1))

    for param_name, param_value in hp1.items():
        print(param_name, param_value)

    print(hp1)

    for param_name, param_value in hp1.items():
        print(param_name, param_value)
    for param_name, param_value in hp2.items():
        print(param_name, param_value)

    print("--")
    print("Hyperparameter Alteration Test")

    print("hp1[1] =", hp1[1])
    print("changing hp1[1]...")
    hp1[1] = ContiniousHyperparameter(1e-6, 1e-0, value=1.0)
    print("hp1[1] =", hp1[1])

    print("--")
    print("print with for loop with indices:")
    for index in range(len(hp1)):
        print(f"hp1[{index}]: {hp1[index]}")
    print("change with for loop with indices:")
    for index in range(len(hp1)):
        hp1[index] = ContiniousHyperparameter(1, 256)
        print(f"hp1[{index}]: {hp1[index]}")

if __name__ == "__main__":
    hyperparameter_math_test()
    print("____________________")
    categorical_hyperparameter_test()
    print("____________________")
    hyperparameter_configuration_test()