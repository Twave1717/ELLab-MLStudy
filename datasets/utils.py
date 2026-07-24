import json
import os

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, items, transform=None, classes=None):
        self.items = items
        self.transform = transform
        self.targets = [label for _, label, _ in items]

        if classes is None:
            class_by_label = {
                label: classname for _, label, classname in items
            }
            classes = [
                class_by_label[label] for label in sorted(class_by_label)
            ]
        self.classes = classes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, label, _ = self.items[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_from_zhou_split(
    root,
    dataset_dir,
    image_dir,
    split_filename,
    train_transform,
    test_transform,
):
    dataset_root = os.path.join(root, dataset_dir)
    image_root = os.path.join(dataset_root, image_dir)
    split_path = os.path.join(dataset_root, split_filename)

    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"Dataset split file not found: {split_path}"
        )

    with open(split_path, "r", encoding="utf-8") as file:
        split = json.load(file)

    def make_items(split_name):
        return [
            (
                os.path.join(image_root, image_path),
                int(label),
                classname,
            )
            for image_path, label, classname in split[split_name]
        ]

    train_items = make_items("train") + make_items("val")
    test_items = make_items("test")

    train_dataset = ImageDataset(train_items, transform=train_transform)
    test_dataset = ImageDataset(
        test_items,
        transform=test_transform,
        classes=train_dataset.classes,
    )
    return train_dataset, test_dataset
