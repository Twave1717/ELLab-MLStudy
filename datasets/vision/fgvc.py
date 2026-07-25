import os
from torchvision.datasets import FGVCAircraft
from .utils import DatasetSpec


dataset_dir = "fgvc_aircraft"


def build_fgvc(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = FGVCAircraft(
        dataset_root,
        split="trainval",
        annotation_level="variant",
        download=True,
        transform=train_transform,
    )
    test_dataset = FGVCAircraft(
        dataset_root,
        split="test",
        annotation_level="variant",
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


DATASET = DatasetSpec(build_fgvc)
