from .utils import build_from_zhou_split


def build_dtd(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="dtd",
        image_dir="images",
        split_filename="split_zhou_DescribableTextures.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
