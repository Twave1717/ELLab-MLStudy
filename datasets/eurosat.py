from .utils import build_from_zhou_split


def build_eurosat(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="eurosat",
        image_dir="2750",
        split_filename="split_zhou_EuroSAT.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
