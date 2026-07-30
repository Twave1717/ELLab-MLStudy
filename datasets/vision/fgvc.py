import os
from torchvision.datasets import FGVCAircraft
from .utils import DatasetSpec


dataset_dir = "fgvc_aircraft"
template = "a photo of a {}, a type of aircraft."


def build_fgvc(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = FGVCAircraft(
        dataset_root,
        split="train",
        annotation_level="variant",
        download=True,
        transform=train_transform,
    )
    val_dataset = FGVCAircraft(
        dataset_root,
        split="val",
        annotation_level="variant",
        download=True,
        transform=test_transform,
    )
    test_dataset = FGVCAircraft(
        dataset_root,
        split="test",
        annotation_level="variant",
        download=True,
        transform=test_transform,
    )
    return train_dataset, val_dataset, test_dataset


DATASET = DatasetSpec(build_fgvc, template)
