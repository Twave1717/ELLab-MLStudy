from dataclasses import dataclass

import torch
from torch.utils.data import Subset


GLOBAL_SEED = 2026


@dataclass(frozen=True)
class DatasetSpec:
    builder: object
    template: str
    crop_size: int = 224

    def build(self, root, train_transform, test_transform):
        datasets = self.builder(root, train_transform, test_transform)
        train = datasets[0]
        classes = [(name,) for name in train.classes]
        for dataset in datasets:
            dataset.classes = classes
            dataset.template = self.template
        return datasets


def _split(dataset, indices):
    subset = Subset(dataset, indices)
    subset.classes = (
        dataset.classes if hasattr(dataset, "classes") else dataset.categories
    )
    return subset


def split_dataset(train_dataset, eval_dataset, test_dataset=None):
    generator = torch.Generator().manual_seed(GLOBAL_SEED)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    if test_dataset is not None:
        split = int(len(indices) * 0.9)
        return (
            _split(train_dataset, indices[:split]),
            _split(eval_dataset, indices[split:]),
            test_dataset,
        )

    train_end = int(len(indices) * 0.7)
    val_end = int(len(indices) * 0.8)
    return (
        _split(train_dataset, indices[:train_end]),
        _split(eval_dataset, indices[train_end:val_end]),
        _split(eval_dataset, indices[val_end:]),
    )
