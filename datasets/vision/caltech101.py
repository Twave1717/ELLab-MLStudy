import os
from torchvision.datasets import Caltech101
from .utils import DatasetSpec, split_dataset


dataset_dir = "caltech-101"
template = "a photo of a {}."


def build_caltech101(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Caltech101(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    eval_test_dataset = Caltech101(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, eval_test_dataset)


DATASET = DatasetSpec(build_caltech101, template)
