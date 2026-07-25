import os
from torchvision.datasets import ImageNet
from .utils import DatasetSpec


dataset_dir = "imagenet"


def build_imagenet(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    # Dataset 파일 수동 다운로드 필요: https://image-net.org/download.php
    train_dataset = ImageNet(
        root=dataset_root,
        split="train",
        transform=train_transform,
    )
    test_dataset = ImageNet(
        root=dataset_root,
        split="val",
        transform=test_transform,
    )
    train_dataset.classes = [names[0] for names in train_dataset.classes]
    test_dataset.classes = train_dataset.classes
    return train_dataset, test_dataset


DATASET = DatasetSpec(build_imagenet)
