from .utils import build_from_zhou_split


def build_caltech101(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="caltech-101",
        image_dir="101_ObjectCategories",
        split_filename="split_zhou_Caltech101.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
