from .utils import build_from_zhou_split


def build_sun397(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="sun397",
        image_dir="SUN397",
        split_filename="split_zhou_SUN397.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
