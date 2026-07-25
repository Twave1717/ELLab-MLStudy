import os
from torchvision.datasets import Imagenette
from .utils import DatasetSpec


dataset_dir = "imagenette"


def build_imagenette(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Imagenette(
        root=dataset_root,
        split="train",
        size="160px",
        download=True,
        transform=train_transform,
    )
    test_dataset = Imagenette(
        root=dataset_root,
        split="val",
        size="160px",
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


DATASET = DatasetSpec(build_imagenette, crop_size=160)
