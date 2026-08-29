from torch.utils.data import DataLoader

from .fewshot import base2new_split, fewshot_subset
from .kaggle import KAGGLE_DATASETS, get_kaggle_dataloader
from .registry import DATASETS
from .transforms import build_transforms


def get_dataloader(
    batch_size,
    dataset_name,
    method="supervised",
    root="data",
    pretrained=False,
    model_name=None,
    shots=None,
    setting="standard",
    kaggle_root="archive/03_kaggle_dataset_and_manifests",
    device="cpu",
):
    if method == "2sfs" and dataset_name in KAGGLE_DATASETS:
        print(f"Using exact Kaggle split: {dataset_name}")
        return get_kaggle_dataloader(
            batch_size, dataset_name, root, kaggle_root, shots, device
        )

    dataset = DATASETS[dataset_name]
    crop_size = 224 if pretrained else dataset.crop_size
    train_transform, test_transform = build_transforms(
        crop_size=crop_size,
        clip=method in {"clip", "2sfs"},
        imagenet=pretrained and (model_name or "").startswith("resnet"),
        two_views=method in {"byol", "moco", "simclr"},
    )

    training_data, validation_data, test_data = dataset.build(
        root,
        train_transform,
        test_transform,
    )
    if setting == "base2new":
        training_data, validation_base_data, validation_new_data, test_base_data, test_new_data = base2new_split(training_data, validation_data, test_data)
    if shots is not None:
        training_data = fewshot_subset(training_data, shots)

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
    )
    if setting == "base2new":
        validation_dataloader = DataLoader(validation_base_data, batch_size=batch_size), DataLoader(validation_new_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_base_data, batch_size=batch_size), DataLoader(test_new_data, batch_size=batch_size)
    else:
        validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
    num_classes = len(training_data.classes)
    return train_dataloader, validation_dataloader, test_dataloader, num_classes
