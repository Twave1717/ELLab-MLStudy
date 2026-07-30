import os
from torchvision.datasets import ImageNet
from .utils import DatasetSpec, split_dataset


dataset_dir = "imagenet"
template = "a photo of a {}."


def build_imagenet(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    # Dataset 파일 수동 다운로드 필요: https://image-net.org/download.php
    train_dataset = ImageNet(
        root=dataset_root,
        split="train",
        transform=train_transform,
    )
    val_dataset = ImageNet(
        root=dataset_root,
        split="train",
        transform=test_transform,
    )
    test_dataset = ImageNet(
        root=dataset_root,
        split="val",
        transform=test_transform,
    )
    train_dataset.classes = [names[0] for names in train_dataset.classes]
    val_dataset.classes = train_dataset.classes
    test_dataset.classes = train_dataset.classes
    return split_dataset(train_dataset, val_dataset, test_dataset)


DATASET = DatasetSpec(build_imagenet, template)
