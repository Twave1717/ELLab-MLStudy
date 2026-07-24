from .utils import build_from_zhou_split


def build_stanford_cars(root, train_transform, test_transform):
    return build_from_zhou_split(
        root=root,
        dataset_dir="stanford_cars",
        image_dir="",
        split_filename="split_zhou_StanfordCars.json",
        train_transform=train_transform,
        test_transform=test_transform,
    )
