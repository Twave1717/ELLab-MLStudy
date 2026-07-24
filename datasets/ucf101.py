from .utils import build_from_zhou_split


def build_ucf101(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="ucf101",
        image_dir="UCF-101-midframes",
        split_filename="split_zhou_UCF101.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
