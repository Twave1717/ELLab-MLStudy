from torchvision.datasets import CIFAR10


def build_cifar10(root, train_transform, test_transform):
    train_dataset = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset
