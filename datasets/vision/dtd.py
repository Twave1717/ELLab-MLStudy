import os
from torchvision.datasets import DTD
from .utils import DatasetSpec, concat_datasets


dataset_dir = "dtd"


def build_dtd(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = DTD(
        dataset_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    val_dataset = DTD(
        dataset_root,
        split="val",
        download=True,
        transform=train_transform,
    )
    test_dataset = DTD(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return concat_datasets(train_dataset, val_dataset), test_dataset


DATASET = DatasetSpec(build_dtd)
