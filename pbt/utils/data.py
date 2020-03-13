import random
import math
import copy
from collections import defaultdict
from typing import Iterable, Dict, Sequence

import numpy as np
import torch
import torchvision
import torch.utils.data
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform

def display_class_balance(labels : Sequence[object]):
    for unique, counts in zip(*np.unique(labels, return_counts=True)):
        print(f"{unique}: {counts} ({counts/len(labels)*100.0:.2f}%)")

def create_subset(dataset, start, end = None) -> Subset:
    end = len(dataset) if not end else end
    return Subset(dataset, list(range(start, end)))

def split(dataset : Dataset, fraction : float) -> (Subset, Subset):
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = round(fraction * dataset_length)
    first_set = create_subset(dataset, 0, first_set_length)
    second_set = create_subset(dataset, first_set_length, dataset_length)
    return first_set, second_set

def random_split(dataset : Dataset, fraction : float, random_state : int = None) -> (Subset, Subset):
    if random_state: torch.manual_seed(random_state)
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = round(fraction * dataset_length)
    second_set_length = dataset_length - first_set_length
    first_set, second_set = torch.utils.data.random_split(
        dataset, (first_set_length, second_set_length))
    first_set = Subset(first_set.dataset, first_set.indices)
    second_set = Subset(second_set.dataset, second_set.indices)
    return first_set, second_set

def stratified_split(dataset : Dataset, labels : Iterable, fraction : float, random_state : int = None, return_labels : bool = False):
    if random_state: random.seed(random_state)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy().tolist()
    first_set_target_length = round(len(labels) * fraction)
    indices_per_label = defaultdict(list)
    label_count = defaultdict(int)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
        label_count[label] += 1
    label_count_sorted = sorted(label_count.items(), key = lambda x: x[1], reverse=True)
    samples_per_label = {label: math.floor(counts * fraction) for label, counts in label_count_sorted}
    size_delta = first_set_target_length - sum(samples_per_label.values())
    first_set_indices = list()
    second_set_indices = list()
    for label, n_samples in samples_per_label.items():
        if size_delta != 0:
            n_samples += 1
            size_delta -= 1
        indices = indices_per_label[label]
        random_indices_sample = random.sample(indices, n_samples)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    random.shuffle(first_set_indices)
    random.shuffle(second_set_indices)
    first_set_inputs = Subset(dataset, first_set_indices)
    second_set_inputs = Subset(dataset, second_set_indices)
    if not return_labels:
        return first_set_inputs, second_set_inputs
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels