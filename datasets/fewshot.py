from math import ceil
from random import Random

from torch.utils.data import Dataset, Subset

from .vision.utils import GLOBAL_SEED


class ClassSubset(Dataset):
    def __init__(self, dataset, classes):
        labels = dataset_labels(dataset)
        label_map = {label: new_label for new_label, label in enumerate(classes)}

        self.dataset = dataset
        self.indices = [index for index, label in enumerate(labels) if label in label_map]
        self.targets = [label_map[labels[index]] for index in self.indices]
        self.classes = [dataset.classes[index] for index in classes]
        self.template = dataset.template

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, _ = self.dataset[self.indices[index]]
        return image, self.targets[index]


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        labels = dataset_labels(dataset.dataset)
        return [labels[index] for index in dataset.indices]

    for attribute in ("targets", "_labels", "y"):
        labels = getattr(dataset, attribute, None)
        if labels is not None:
            return labels

    return [label for _, label in dataset._samples]


def fewshot_subset(dataset, shots):
    indices_by_class = {label: [] for label in range(len(dataset.classes))}
    for index, label in enumerate(dataset_labels(dataset)):
        indices_by_class[label].append(index)

    rng = Random(GLOBAL_SEED)
    selected_indices = []
    for candidates in indices_by_class.values():
        if len(candidates) >= shots:
            selected_indices.extend(rng.sample(candidates, shots))
        else:
            selected_indices.extend(rng.choices(candidates, k=shots))

    subset = Subset(dataset, selected_indices)
    subset.classes = dataset.classes
    subset.template = dataset.template
    return subset


def base2new_split(train, validation, test):
    split = ceil(len(train.classes) / 2)
    base_classes = range(split)
    new_classes = range(split, len(train.classes))
    return (
        ClassSubset(train, base_classes),
        ClassSubset(validation, base_classes),
        ClassSubset(validation, new_classes),
        ClassSubset(test, base_classes),
        ClassSubset(test, new_classes)
    )
