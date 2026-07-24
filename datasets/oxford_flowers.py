from .utils import build_from_zhou_split


def build_oxford_flowers(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="oxford_flowers",
        image_dir="jpg",
        split_filename="split_zhou_OxfordFlowers.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
