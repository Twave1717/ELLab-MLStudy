from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomResizedCrop, Normalize

from .registry import get_dataset_config


class TwoViewTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def get_dataloader(batch_size, dataset_name, method="supervised", root="data"):
    dataset_config = get_dataset_config(dataset_name)
    crop_size = dataset_config.crop_size

    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomResizedCrop(crop_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = Compose([
        RandomResizedCrop(crop_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if method not in ("supervised", "rotnet", "rotnet_eval"):
        train_transform = TwoViewTransform(train_transform)
        test_transform = TwoViewTransform(test_transform)

    training_data, test_data = dataset_config.builder(
        root,
        train_transform,
        test_transform,
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    num_classes = len(training_data.classes)

    for X, y in test_dataloader:
        if method in {"supervised", "rotnet", "rotnet_eval"}:
            print(f"Shape of X [B, C, H, W]: {X.shape}")
        else:
            print(f"Shape of X1 [B, C, H, W]: {X[0].shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader, num_classes
