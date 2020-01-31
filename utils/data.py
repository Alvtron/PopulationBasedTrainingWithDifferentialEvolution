import random
import math
import torch
import torchvision
import copy
import numpy as np
from torch.utils.data import Dataset, Subset
from hyperparameters import Hyperparameter, Hyperparameters
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from collections import defaultdict
from typing import Iterable, Dict, Sequence

class AdaptiveDataset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset : Dataset,
            prefix_transform : list = None, prefix_target_transform : list = None,
            suffix_transform : list = None, suffix_target_transform : list = None):
        super().__init__()
        self.dataset = dataset
        self.prefix_transform = prefix_transform
        self.suffix_transform = suffix_transform
        self.prefix_target_transform = prefix_target_transform
        self.suffix_target_transform = suffix_target_transform
        self.transform = None
        self.update(None)

    def __getitem__(self, index):
        image, labels = self.dataset[index]
        return self.transform(image, labels)

    def __len__(self):
        return len(self.dataset)

    def copy(self):
        copied = copy.copy(self)
        copied.prefix_transform = copy.deepcopy(self.prefix_transform)
        copied.suffix_transform = copy.deepcopy(self.suffix_transform)
        copied.prefix_target_transform = copy.deepcopy(self.prefix_target_transform)
        copied.suffix_target_transform = copy.deepcopy(self.suffix_target_transform)
        copied.transform = copy.deepcopy(self.transform)
        return copied

    def split(self, fraction : float):
        subset = self.copy()
        subset.dataset = split(subset.dataset, fraction)
        return subset

    def subset(self, indices : Sequence[int]):
        subset = self.copy()
        subset.dataset = Subset(self.dataset, indices)
        return Subset

    @staticmethod
    def create_hyper_parameters(include : Sequence[str] = None) -> Dict[str, Hyperparameter]:
        hparams = {
            'brightness' : Hyperparameter(0.0, 1.0),
            'contrast' : Hyperparameter(0.0, 1.0),
            'saturation' : Hyperparameter(0.0, 1.0),
            'hue' : Hyperparameter(0.0, 1.0),
            'rotate' : Hyperparameter(0, 180),
            'translate_horizontal' : Hyperparameter(0.0, 1.0),
            'translate_vertical' : Hyperparameter(0.0, 1.0),
            'scale_min' : Hyperparameter(0.5, 1.5),
            'scale_max' : Hyperparameter(0.5, 1.5),
            'shear' : Hyperparameter(0, 90),
            'perspective' : Hyperparameter(0.0, 1.0),
            'vertical_flip' : Hyperparameter(0.0, 1.0),
            'horizontal_flip' : Hyperparameter(0.0, 1.0)}
        if include:
            exclude = [param_name for param_name in hparams if param_name not in include]
            for param_name in exclude:
                del hparams[param_name]
        return hparams

    def update(self, hparams : Dict[str, Hyperparameter] = None):
        transforms = list()
        target_transforms = list()
        # add original transform at the end
        if self.prefix_transform:
            transforms.extend(self.prefix_transform)
        if self.prefix_target_transform:
            target_transforms.extend(self.prefix_transform)
        if hparams:
            # random color jitter
            color_jitter_transforms = ['brightness', 'contrast', 'saturation', 'hue']
            if any(x in hparams for x in color_jitter_transforms):
                transforms.append(torchvision.transforms.ColorJitter(
                    brightness=hparams['brightness'].value if 'brightness' in hparams else None,
                    contrast=hparams['contrast'].value if 'contrast' in hparams else None,
                    saturation=hparams['saturation'].value if 'saturation' in hparams else None,
                    hue=hparams['hue'].value if 'hue' in hparams else None))
            # random horizontal flip
            if 'horizontal_flip' in hparams:
                transforms.append(torchvision.transforms.RandomHorizontalFlip(
                    p=hparams['horizontal_flip'].value))
            # random vertical flip
            if 'vertical_flip' in hparams:
                transforms.append(torchvision.transforms.RandomVerticalFlip(
                    p=hparams['vertical_flip'].value))
            # random perspective
            if 'perspective' in hparams:
                transforms.append(torchvision.transforms.RandomPerspective(
                    distortion_scale=hparams['perspective'].value,
                    p=1.0))
            # random affine
            affine_transforms = ['rotate', 'translate_horizontal', 'translate_vertical', 'scale_min', 'scale_max', 'shear']
            if any(x in hparams for x in affine_transforms):
                transforms.append(torchvision.transforms.RandomAffine(
                    degrees=(hparams['rotate'].value if 'rotate' in hparams else 0),
                    translate=(
                        hparams['translate_horizontal'].value,
                        hparams['translate_vertical'].value
                    ) if 'translate_horizontal' in hparams and 'translate_horizontal' in hparams else None,
                    scale=(
                        hparams['scale_min'].value,
                        hparams['scale_max'].value
                    ) if 'scale_min' in hparams and 'scale_max' in hparams else None,
                    shear=hparams['shear'].value if 'shear' in hparams else 0,
                    fillcolor=0))
        # add original target transform at the end
        if self.suffix_transform:
            transforms.extend(self.suffix_transform)
        if self.suffix_target_transform:
            target_transforms.extend(self.suffix_target_transform)
        transform_comp = torchvision.transforms.Compose(transforms) if transforms else None
        target_transform_comp = torchvision.transforms.Compose(target_transforms) if target_transforms else None
        self.transform = StandardTransform(transform_comp, target_transform_comp)

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
    first_set, second_set = random_split(
        dataset, (first_set_length, second_set_length))
    first_set = Subset(first_set.dataset, first_set.indices)
    second_set = Subset(second_set.dataset, second_set.indices)
    return first_set, second_set

def stratified_split(dataset : Dataset, labels : Iterable, fraction : float, random_state : int = None, verbose : bool = False):
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
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    if verbose:
        print("first_set:")
        display_class_balance(first_set_labels)
        print(f"Length: {len(first_set_labels)}")
        print("second_set:")
        display_class_balance(second_set_labels)
        print(f"Length: {len(second_set_labels)}")
        kek = first_set_labels + second_set_labels
        print(f"Is distinct? {len(kek) != len(set(kek))}")
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels