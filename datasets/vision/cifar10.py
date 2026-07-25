import os
from torchvision.datasets import CIFAR10
from .utils import DatasetSpec


dataset_dir = "cifar10"


def build_cifar10(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = CIFAR10(
        root=dataset_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


DATASET = DatasetSpec(build_cifar10, crop_size=32)
