import os
from torchvision.datasets import StanfordCars
from .utils import DatasetSpec, split_dataset


dataset_dir = "stanford_cars"
template = "a photo of a {}."


def build_stanford_cars(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = StanfordCars(
        dataset_root,
        split="train",
        download=True,
        transform=train_transform,
    )
    val_dataset = StanfordCars(
        dataset_root,
        split="train",
        download=True,
        transform=test_transform,
    )
    test_dataset = StanfordCars(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_stanford_cars, template)
