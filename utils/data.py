import random
import math
import torch.utils.data
import torchvision
import torchvision.transforms
from copy import copy
from hyperparameters import Hyperparameter, Hyperparameters
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform
from collections import defaultdict
from typing import Iterable, Dict, Sequence

def get_compose(dataset):
    if type(dataset) == torch.utils.data.Subset:
        return dataset.dataset.transforms
    elif type(dataset) == VisionDataset:
        return dataset.transforms
    else:
        return None

def set_transforms(dataset, transforms):
    if type(dataset) == VisionDataset:
        dataset.transforms = transforms
    if type(dataset) == torch.utils.data.Subset:
        dataset.dataset.transforms = transforms
    else:
        raise TypeError()

class AdaptiveDataset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset : torch.utils.data.Dataset, prefix_transform = None, prefix_target_transform = None, suffix_transform = None, suffix_target_transform = None, indices : Iterable[int] = None):
        self.dataset = dataset
        self.indices = list(indices)
        self.prefix_transform = prefix_transform
        self.suffix_transform = suffix_transform
        self.prefix_target_transform = prefix_target_transform
        self.suffix_target_transform = suffix_target_transform
        self.transform = None
        self.update(None)

    def __getitem__(self, idx):
        image, labels = self.dataset[self.indices[idx]] if self.indices else self.dataset[idx]
        return self.transform(image, labels)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def create_hyper_parameters(include : Sequence[str] = None) -> Dict[str, Hyperparameter]:
        hparams = {
            'brightness' : Hyperparameter(0.0, 1.0),
            'contrast' : Hyperparameter(0.0, 1.0),
            'saturation' : Hyperparameter(0.0, 1.0),
            'hue' : Hyperparameter(0.0, 1.0),
            'degrees' : Hyperparameter(0, 180),
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
            affine_transforms = ['degrees', 'translate_horizontal', 'translate_vertical', 'scale_min', 'scale_max', 'shear']
            if any(x in hparams for x in affine_transforms):
                transforms.append(torchvision.transforms.RandomAffine(
                    degrees=(hparams['degrees'].value if 'degrees' in hparams else 0),
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
        transform_comp = torchvision.transforms.Compose(transforms)
        target_transform_comp = torchvision.transforms.Compose(target_transforms)
        self.transform = StandardTransform(transform_comp, target_transform_comp)

def create_subset(dataset, start, end = None) -> AdaptiveDataset:
    end = len(dataset) if not end else end
    if isinstance(dataset, AdaptiveDataset):
        subset = copy(dataset)
        subset.indices = range(start, end)
        return subset
    else:
        return AdaptiveDataset(dataset, indices=range(start, end))

def split(dataset : torch.utils.data.Dataset, fraction : float) -> (AdaptiveDataset, AdaptiveDataset):
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = round(fraction * dataset_length)
    first_set = create_subset(dataset, 0, first_set_length)
    second_set = create_subset(dataset, first_set_length, dataset_length)
    return first_set, second_set

def random_split(dataset : torch.utils.data.Dataset, fraction : float, random_state : int = None) -> (AdaptiveDataset, AdaptiveDataset):
    if random_state: torch.manual_seed(random_state)
    assert 0.0 <= fraction <= 1.0, f"The provided fraction must be between 0.0 and 1.0!"
    dataset_length = len(dataset)
    first_set_length = round(fraction * dataset_length)
    second_set_length = dataset_length - first_set_length
    first_set, second_set = torch.utils.data.random_split(
        dataset, (first_set_length, second_set_length))
    first_set = AdaptiveDataset(first_set.dataset, indices=first_set.indices)
    second_set = AdaptiveDataset(second_set.dataset, indices=second_set.indices)
    return first_set, second_set

def stratified_split(dataset : torch.utils.data.Dataset, labels : Iterable, fraction : float, random_state : int = None):
    if random_state: random.seed(random_state)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = math.floor(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = AdaptiveDataset(dataset, indices=first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = AdaptiveDataset(dataset, indices=second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels