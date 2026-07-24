from .utils import build_from_zhou_split


def build_oxford_pets(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="oxford_pets",
        image_dir="images",
        split_filename="split_zhou_OxfordPets.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
