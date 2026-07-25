from torch.utils.data import DataLoader

from .registry import DATASETS
from .transforms import build_transforms


def get_dataloader(
    batch_size,
    dataset_name,
    method="supervised",
    root="data",
    pretrained=False,
    model_name=None,
):
    dataset = DATASETS[dataset_name]
    crop_size = 224 if pretrained else dataset.crop_size
    train_transform, test_transform = build_transforms(
        crop_size=crop_size,
        clip=method == "clip" and not pretrained,
        imagenet=pretrained and (model_name or "").startswith("resnet"),
        two_views=method in {"byol", "moco", "simclr"},
    )

    training_data, validation_data, test_data = dataset.build(
        root,
        train_transform,
        test_transform,
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
    )
    num_classes = len(training_data.classes)
    return train_dataloader, validation_dataloader, test_dataloader, num_classes
