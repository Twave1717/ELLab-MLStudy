import os
from torchvision.datasets import Food101
from .utils import DatasetSpec, split_dataset


dataset_dir = "food-101"
template = "a photo of {}, a type of food."


def build_food101(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Food101(
        dataset_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    val_dataset = Food101(
        dataset_root,
        split="train",
        download=True,
        transform=test_transform,
    )
    test_dataset = Food101(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_food101, template)
