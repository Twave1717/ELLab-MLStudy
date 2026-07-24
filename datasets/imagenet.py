from torchvision.datasets import ImageNet


def build_imagenet(root, train_transform, test_transform):
    # Dataset 파일 수동 다운로드 필요: https://image-net.org/download.php
    train_dataset = ImageNet(
        root=root,
        split="train",
        transform=train_transform,
    )
    test_dataset = ImageNet(
        root=root,
        split="val",
        transform=test_transform,
    )
    return train_dataset, test_dataset
