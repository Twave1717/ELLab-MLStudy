import os
from torchvision.datasets import Imagenette
from .utils import DatasetSpec, split_dataset


dataset_dir = "imagenette"
template = "a photo of a {}."


def build_imagenette(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = Imagenette(
        root=dataset_root,
        split="train",
        size="160px",
        download=True,
        transform=train_transform,
    )
    val_dataset = Imagenette(
        root=dataset_root,
        split="train",
        size="160px",
        download=True,
        transform=test_transform,
    )
    test_dataset = Imagenette(
        root=dataset_root,
        split="val",
        size="160px",
        download=True,
        transform=test_transform,
    )
    train_dataset.classes = [names[0] for names in train_dataset.classes]
    val_dataset.classes = train_dataset.classes
    test_dataset.classes = train_dataset.classes
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_imagenette, template, crop_size=160)
