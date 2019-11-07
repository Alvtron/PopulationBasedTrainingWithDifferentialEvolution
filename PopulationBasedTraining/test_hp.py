import random
from hyperparameters import Hyperparameter

categorical = Hyperparameter("A","B","C","D","E","F","G","H","I", is_categorical = True)

print("sampling uniform...")
for i in range(3):
    categorical.sample_uniform()
    print(f"{categorical.normalized()}: {categorical.value()}")

print("perturbing...")
for i in range(3):
    random_value = random.choice((0.8, 1.1))
    categorical *= random_value
    print(f"{categorical.normalized()}: {categorical.value()}")

print("--")
print(f"Normalized: {categorical.normalized()}")
print(f"Value: {categorical.value()}")
print(f"Lower bound: {categorical.get_lower_bound()}")
print(f"Upper bound: {categorical.get_upper_bound()}")
print("--")
print(f"Normalized value {0.30} gives me value {categorical.get_value(0.30)}")
for value in categorical._search_space:
    print(f"Value {value} gives me normalized value {categorical.get_normalized_value(value)}")