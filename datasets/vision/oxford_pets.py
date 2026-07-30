import os
from torchvision.datasets import OxfordIIITPet
from .utils import DatasetSpec, split_dataset


dataset_dir = "oxford_pets"
template = "a photo of a {}, a type of pet."


def build_oxford_pets(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = OxfordIIITPet(
        dataset_root,
        split="trainval",
        download=True,
        transform=train_transform,
    )
    val_dataset = OxfordIIITPet(
        dataset_root,
        split="trainval",
        download=True,
        transform=test_transform,
    )
    test_dataset = OxfordIIITPet(
        dataset_root,
        split="test",
        download=True,
        transform=test_transform,
    )
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_oxford_pets, template)
