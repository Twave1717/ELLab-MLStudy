import os
from torchvision.datasets import StanfordCars
from .utils import DatasetSpec


dataset_dir = "stanford_cars"


def build_stanford_cars(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = StanfordCars(
        dataset_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    test_dataset = StanfordCars(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


DATASET = DatasetSpec(build_stanford_cars)
