from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from notebook.utils import data as kaggle_data

from .vision.utils import GLOBAL_SEED


KAGGLE_DATASETS = kaggle_data.DATASETS


def get_kaggle_dataloader(
    batch_size,
    dataset_name,
    root="data",
    kaggle_root="archive/03_kaggle_dataset_and_manifests",
    shots=16,
    device="cpu",
):
    repo_root = Path(__file__).resolve().parents[1]
    config = SimpleNamespace(
        repo_root=repo_root,
        data_root=(repo_root / root).resolve(),
        kaggle_root=(repo_root / kaggle_root).resolve(),
        shots=shots,
        datasets=(dataset_name,),
    )
    train_transform, eval_transform = kaggle_data.clip_transforms()
    train = kaggle_data.load_train_dataset(config, dataset_name, train_transform)
    validation = kaggle_data.load_validation_datasets(
        config, dataset_name, eval_transform
    )
    test_base, test_novel, _ = kaggle_data.load_test_datasets_for_evaluation(
        config, dataset_name, eval_transform
    )

    def loader(dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=torch.device(device).type == "cuda",
            generator=(
                torch.Generator().manual_seed(GLOBAL_SEED + 101)
                if shuffle
                else None
            ),
        )

    return (
        loader(train, shuffle=True),
        tuple(loader(dataset) for dataset in validation),
        (loader(test_base), loader(test_novel)),
        len(train.classes),
    )
