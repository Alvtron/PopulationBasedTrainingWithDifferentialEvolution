import random
import math
import copy
from collections import defaultdict
from typing import Iterable, Dict, Sequence

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.vision import StandardTransform

from .hyperparameters import ContiniousHyperparameter, Hyperparameters

class Datasets(object):
    def __init__(self, train_data, eval_data, test_data):
        self.train = train_data
        self.eval = eval_data
        self.test = test_data

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
    def create_hyper_parameters(include : Sequence[str] = None) -> Dict[str, ContiniousHyperparameter]:
        hparams = {
            'brightness' : ContiniousHyperparameter(0.0, 1.0),
            'contrast' : ContiniousHyperparameter(0.0, 1.0),
            'saturation' : ContiniousHyperparameter(0.0, 1.0),
            'hue' : ContiniousHyperparameter(0.0, 1.0),
            'rotate' : ContiniousHyperparameter(0, 180),
            'translate_horizontal' : ContiniousHyperparameter(0.0, 1.0),
            'translate_vertical' : ContiniousHyperparameter(0.0, 1.0),
            'scale_min' : ContiniousHyperparameter(0.5, 1.5),
            'scale_max' : ContiniousHyperparameter(0.5, 1.5),
            'shear' : ContiniousHyperparameter(0, 90),
            'perspective' : ContiniousHyperparameter(0.0, 1.0),
            'vertical_flip' : ContiniousHyperparameter(0.0, 1.0),
            'horizontal_flip' : ContiniousHyperparameter(0.0, 1.0)}
        if include:
            exclude = [param_name for param_name in hparams if param_name not in include]
            for param_name in exclude:
                del hparams[param_name]
        return hparams

    def update(self, hparams : Dict[str, ContiniousHyperparameter] = None):
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