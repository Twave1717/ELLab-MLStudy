"""Kaggle manifests and datasets for the 2SFS notebooks."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import build_transforms


class Config(Protocol):
    repo_root: Path
    shots: int
    datasets: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    prompt: str
    image_root: str
    validation_manifest: str
    validation_format: str
    test_counts: tuple[int, int]


DATASET_SPECS = {
    "eurosat": DatasetSpec(
        "a centered satellite photo of {}.",
        "data/eurosat/eurosat/2750",
        "data/2sfs_splits/coop/split_zhou_EuroSAT.json",
        "json",
        (4200, 3900),
    ),
    "fgvc_aircraft": DatasetSpec(
        "a photo of a {}, a type of aircraft.",
        "data/fgvc_aircraft/fgvc-aircraft-2013b/data/images",
        "data/fgvc_aircraft/fgvc-aircraft-2013b/data/images_variant_val.txt",
        "text",
        (1666, 1667),
    ),
    "dtd": DatasetSpec(
        "{} texture.",
        "data/dtd/dtd/dtd/images",
        "data/2sfs_splits/coop/split_zhou_DescribableTextures.json",
        "json",
        (862, 828),  # Two Base images duplicate train images.
    ),
}
DATASETS = tuple(DATASET_SPECS)
EXPECTED_TEST_COUNTS = {
    name: dict(zip(("base", "novel"), spec.test_counts))
    for name, spec in DATASET_SPECS.items()
}


@dataclass(frozen=True)
class ClassEntry:
    label: int
    key: str
    name: str
    split: str


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class ManifestDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ImageSample],
        catalog: Sequence[ClassEntry],
        template: str,
        transform: Any,
    ) -> None:
        self.samples = tuple(samples)
        self.catalog = tuple(catalog)
        self.classes = [(entry.name,) for entry in self.catalog]
        self.template = template
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_catalog(config: Config, dataset: str) -> tuple[ClassEntry, ...]:
    rows = [
        row
        for row in read_csv(config.repo_root / "kaggle/public/classes.csv")
        if row.get("dataset") == dataset
    ]
    catalog = tuple(
        ClassEntry(index, row["class_key"], row["class_name"], row["class_split"])
        for index, row in enumerate(rows)
    )
    keys = [entry.key for entry in catalog]
    if not catalog or len(keys) != len(set(keys)):
        raise ValueError(f"Missing or duplicate classes for {dataset}")
    if {entry.split for entry in catalog} != {"base", "novel"}:
        raise ValueError(f"Invalid Base/Novel split for {dataset}")
    return catalog


def _local_catalog(catalog: Sequence[ClassEntry], split: str) -> tuple[ClassEntry, ...]:
    return tuple(
        ClassEntry(label, entry.key, entry.name, entry.split)
        for label, entry in enumerate(item for item in catalog if item.split == split)
    )


def _safe_path(root: Path, path_text: str) -> Path:
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {path_text!r}")
    root = root.resolve()
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes root: {path_text!r}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def clip_transforms() -> tuple[Any, Any]:
    return build_transforms(crop_size=224, clip=True)


def load_train_dataset(
    config: Config, dataset: str, transform: Any | None = None
) -> ManifestDataset:
    catalog = _local_catalog(load_catalog(config, dataset), "base")
    by_key = {entry.key: entry for entry in catalog}
    manifest = config.repo_root / f"kaggle/public/train_{config.shots}shot.csv"
    rows = [row for row in read_csv(manifest) if row.get("dataset") == dataset]
    public_root = config.repo_root / "kaggle/public"
    samples: list[ImageSample] = []
    counts: Counter[int] = Counter()
    for row in rows:
        try:
            entry = by_key[row["class_key"]]
        except KeyError as exc:
            raise ValueError(f"Unknown class in {manifest}: {row}") from exc
        samples.append(
            ImageSample(_safe_path(public_root, row["image_path"]), entry.label)
        )
        counts[entry.label] += 1
    expected = {entry.label: config.shots for entry in catalog}
    if {label: counts[label] for label in expected} != expected:
        raise ValueError(f"{dataset} is not exact {config.shots}-shot")
    if transform is None:
        transform, _ = clip_transforms()
    return ManifestDataset(samples, catalog, DATASET_SPECS[dataset].prompt, transform)


def validation_manifest_path(config: Config, dataset: str) -> Path:
    return config.repo_root / DATASET_SPECS[dataset].validation_manifest


def _validation_samples(
    config: Config, dataset: str, catalog: Sequence[ClassEntry]
) -> list[tuple[Path, ClassEntry]]:
    spec = DATASET_SPECS[dataset]
    manifest = validation_manifest_path(config, dataset)
    root = config.repo_root / spec.image_root
    samples: list[tuple[Path, ClassEntry]] = []
    if spec.validation_format == "json":
        for relative, label, class_name in _read_json(manifest)["val"]:
            entry = catalog[int(label)]
            if class_name != entry.name:
                raise ValueError(
                    f"Validation class mismatch for {dataset}: {class_name!r}"
                )
            samples.append((_safe_path(root, str(relative)), entry))
    else:
        by_name = {entry.name: entry for entry in catalog}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line:
                image_id, class_name = line.split(" ", 1)
                samples.append(
                    (_safe_path(root, f"{image_id}.jpg"), by_name[class_name])
                )
    paths = [path for path, _ in samples]
    labels = {entry.label for _, entry in samples}
    if not samples or len(paths) != len(set(paths)) or len(labels) != len(catalog):
        raise ValueError(f"Incomplete or duplicate validation manifest: {manifest}")
    return samples


def _split_datasets(
    dataset: str,
    catalog: Sequence[ClassEntry],
    samples: Sequence[tuple[Path, ClassEntry]],
    transform: Any,
) -> tuple[ManifestDataset, ManifestDataset]:
    catalogs = {split: _local_catalog(catalog, split) for split in ("base", "novel")}
    labels = {entry.key: entry.label for group in catalogs.values() for entry in group}
    grouped: dict[str, list[ImageSample]] = {"base": [], "novel": []}
    for path, entry in samples:
        grouped[entry.split].append(ImageSample(path, labels[entry.key]))
    return tuple(
        ManifestDataset(
            grouped[split], catalogs[split], DATASET_SPECS[dataset].prompt, transform
        )
        for split in ("base", "novel")
    )


def load_validation_datasets(
    config: Config, dataset: str, transform: Any | None = None
) -> tuple[ManifestDataset, ManifestDataset]:
    catalog = load_catalog(config, dataset)
    if transform is None:
        _, transform = clip_transforms()
    return _split_datasets(
        dataset, catalog, _validation_samples(config, dataset, catalog), transform
    )


def load_test_datasets_for_evaluation(
    config: Config, dataset: str, transform: Any | None = None
) -> tuple[ManifestDataset, ManifestDataset, dict[str, tuple[str, ...]]]:
    catalog = load_catalog(config, dataset)
    by_key = {entry.key: entry for entry in catalog}
    public_root = config.repo_root / "kaggle/public"
    test = [
        row for row in read_csv(public_root / "test.csv") if row["dataset"] == dataset
    ]
    solution = [
        row
        for row in read_csv(config.repo_root / "kaggle/host_only/solution.csv")
        if row["dataset"] == dataset
    ]
    truth = {row["id"]: row for row in solution}
    ids = [row["id"] for row in test]
    if (
        len(truth) != len(solution)
        or len(set(ids)) != len(ids)
        or set(ids) != set(truth)
    ):
        raise ValueError(f"Test/solution ID mismatch for {dataset}")

    samples: list[tuple[Path, ClassEntry]] = []
    ids_by_split: dict[str, list[str]] = {"base": [], "novel": []}
    for row in test:
        target = truth[row["id"]]
        if target["eval_split"] != row["eval_split"]:
            raise ValueError(f"Test split mismatch for ID {row['id']}")
        entry = by_key[target["target"]]
        if entry.split != target["eval_split"]:
            raise ValueError(f"Class/split mismatch for ID {row['id']}")
        samples.append((_safe_path(public_root, row["image_path"]), entry))
        ids_by_split[entry.split].append(row["id"])
    counts = {split: len(ids_by_split[split]) for split in ("base", "novel")}
    if counts != EXPECTED_TEST_COUNTS[dataset]:
        raise ValueError(f"Unexpected corrected test counts for {dataset}: {counts}")
    if transform is None:
        _, transform = clip_transforms()
    base, novel = _split_datasets(dataset, catalog, samples, transform)
    return base, novel, {split: tuple(ids_by_split[split]) for split in ids_by_split}


def describe_breakpoint_inputs(config: Config) -> list[dict[str, Any]]:
    train = read_csv(config.repo_root / f"kaggle/public/train_{config.shots}shot.csv")
    summaries = []
    for dataset in config.datasets:
        catalog = load_catalog(config, dataset)
        validation = _validation_samples(config, dataset, catalog)
        summaries.append(
            {
                "dataset": dataset,
                "classes": len(catalog),
                "base_classes": sum(entry.split == "base" for entry in catalog),
                "novel_classes": sum(entry.split == "novel" for entry in catalog),
                "train_images": sum(row.get("dataset") == dataset for row in train),
                "validation_images": len(validation),
                "validation_manifest": str(validation_manifest_path(config, dataset)),
            }
        )
    return summaries
