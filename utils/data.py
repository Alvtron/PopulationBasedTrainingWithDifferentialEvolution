import random
import math
import torch.utils.data
from collections import defaultdict

def random_split(dataset : torch.utils.data.Dataset, fraction):
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = math.floor(fraction * dataset_length)
    second_set_length = dataset_length - first_set_length
    first_set, second_set = torch.utils.data.random_split(
        dataset, (first_set_length, second_set_length))
    return first_set, second_set

def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels