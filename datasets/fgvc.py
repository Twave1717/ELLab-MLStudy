import os

from .utils import ImageDataset


def build_fgvc(root, train_transform, test_transform):
    dataset_root = os.path.join(root, "fgvc_aircraft")
    image_root = os.path.join(dataset_root, "images")

    with open(
        os.path.join(dataset_root, "variants.txt"),
        "r",
        encoding="utf-8",
    ) as file:
        classes = [line.strip() for line in file if line.strip()]

    label_by_class = {
        classname: label for label, classname in enumerate(classes)
    }

    def read_split(filename):
        split_path = os.path.join(dataset_root, filename)
        with open(split_path, "r", encoding="utf-8") as file:
            items = []
            for line in file:
                image_name, classname = line.strip().split(" ", maxsplit=1)
                items.append(
                    (
                        os.path.join(image_root, f"{image_name}.jpg"),
                        label_by_class[classname],
                        classname,
                    )
                )
        return items

    train_items = read_split("images_variant_train.txt")
    train_items += read_split("images_variant_val.txt")
    test_items = read_split("images_variant_test.txt")

    train_dataset = ImageDataset(
        train_items,
        transform=train_transform,
        classes=classes,
    )
    test_dataset = ImageDataset(
        test_items,
        transform=test_transform,
        classes=classes,
    )
    return train_dataset, test_dataset
