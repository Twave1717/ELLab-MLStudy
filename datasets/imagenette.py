from torchvision.datasets import Imagenette


def build_imagenette(root, train_transform, test_transform):
    train_dataset = Imagenette(
        root=root,
        split="train",
        size="160px",
        download=True,
        transform=train_transform,
    )
    test_dataset = Imagenette(
        root=root,
        split="val",
        size="160px",
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset
