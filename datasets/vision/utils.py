from dataclasses import dataclass
import torch
from torch.utils.data import ConcatDataset, Subset


@dataclass(frozen=True)
class DatasetSpec:
    builder: object
    crop_size: int = 224

    def build(self, root, train_transform, test_transform):
        train, test = self.builder(
            root,
            train_transform,
            test_transform,
        )
        classes = [(name,) for name in train.classes]
        train.classes = classes
        test.classes = classes
        return train, test


def concat_datasets(*datasets):
    combined = ConcatDataset(datasets)
    combined.classes = datasets[0].classes
    return combined


def random_split_datasets(train_dataset, test_dataset, train_ratio=0.7):
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    split = int(len(indices) * train_ratio)

    train = Subset(train_dataset, indices[:split])
    test = Subset(test_dataset, indices[split:])
    classes = getattr(train_dataset, "classes", None)
    if classes is None:
        classes = train_dataset.categories
    train.classes = classes
    test.classes = classes
    return train, test
