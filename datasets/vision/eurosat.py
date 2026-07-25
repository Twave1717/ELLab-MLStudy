import os
from torchvision.datasets import EuroSAT
from .utils import DatasetSpec, split_dataset


dataset_dir = "eurosat"
template = "a centered satellite photo of {}."
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
    eval_test_dataset = EuroSAT(
        dataset_root,
        download=True,
        transform=test_transform,
    )
    train_dataset.classes = [
        NEW_CNAMES.get(name, name) for name in train_dataset.classes
    ]
    eval_test_dataset.classes = train_dataset.classes
    return split_dataset(train_dataset, eval_test_dataset)


DATASET = DatasetSpec(build_eurosat, template)
