import os
from torchvision.datasets import EuroSAT
from .utils import DatasetSpec, random_split_datasets


dataset_dir = "eurosat"
NEW_CNAMES = {
    "AnnualCrop": "Annual Crop Land",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Vegetation Land",
    "Highway": "Highway or Road",
    "Industrial": "Industrial Buildings",
    "Pasture": "Pasture Land",
    "PermanentCrop": "Permanent Crop Land",
    "Residential": "Residential Buildings",
    "River": "River",
    "SeaLake": "Sea or Lake",
}


def build_eurosat(root, train_transform, test_transform):
    dataset_root = os.path.join(root, dataset_dir)
    train_dataset = EuroSAT(
        dataset_root,
        download=True,
        transform=train_transform,
    )
    test_dataset = EuroSAT(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    train_dataset.classes = [
        NEW_CNAMES.get(name, name) for name in train_dataset.classes
    ]
    test_dataset.classes = train_dataset.classes
    return random_split_datasets(train_dataset, test_dataset)


DATASET = DatasetSpec(build_eurosat)
