import os
from torchvision.datasets import Flowers102
from .utils import DatasetSpec, concat_datasets


dataset_dir = "oxford_flowers"


def build_oxford_flowers(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Flowers102(
        dataset_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    val_dataset = Flowers102(
        dataset_root,
        split="val",
        download=True,
        transform=train_transform,
    )
    test_dataset = Flowers102(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return concat_datasets(train_dataset, val_dataset), test_dataset


DATASET = DatasetSpec(build_oxford_flowers)
