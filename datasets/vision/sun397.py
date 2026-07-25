import os
from torchvision.datasets import SUN397
from .utils import DatasetSpec, random_split_datasets


dataset_dir = "sun397"


def build_sun397(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = SUN397(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    test_dataset = SUN397(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    return random_split_datasets(train_dataset, test_dataset)


DATASET = DatasetSpec(build_sun397)
