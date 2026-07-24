from .utils import build_from_zhou_split


def build_food101(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="food-101",
        image_dir="images",
        split_filename="split_zhou_Food101.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
