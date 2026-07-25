import os
from torchvision.datasets import Caltech101
from .utils import DatasetSpec, random_split_datasets


dataset_dir = "caltech-101"


def build_caltech101(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Caltech101(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    test_dataset = Caltech101(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    return random_split_datasets(train_dataset, test_dataset)


DATASET = DatasetSpec(build_caltech101)
