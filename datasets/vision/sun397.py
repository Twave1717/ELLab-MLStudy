import os
from torchvision.datasets import SUN397
from .utils import DatasetSpec, split_dataset


dataset_dir = "sun397"
template = "a photo of a {}."


def build_sun397(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = SUN397(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    eval_test_dataset = SUN397(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, eval_test_dataset)


DATASET = DatasetSpec(build_sun397, template)
