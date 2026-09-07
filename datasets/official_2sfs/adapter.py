import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from ..transforms import build_transforms

from .upstream import build_dataset


UPSTREAM_COMMIT = "64ac143c7d22803bfeeaadfe42f570220ab29b06"
OFFICIAL_SPLIT_SEEDS = (1, 2, 3)
OFFICIAL_2SFS_DATASETS = (
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
)

_DATASET_DIRS = {
    "caltech101": "caltech-101",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "fgvc": "fgvc_aircraft",
    "food101": "food-101",
    "oxford_flowers": "oxford_flowers",
    "oxford_pets": "oxford_pets",
    "stanford_cars": "stanford_cars",
    "sun397": "sun397",
    "ucf101": "ucf101",
}

_IMAGE_DIRS = {
    "caltech101": "101_ObjectCategories",
    "dtd": "images",
    "eurosat": "2750",
    "fgvc": "images",
    "food101": "images",
    "oxford_flowers": "jpg",
    "oxford_pets": "images",
    "stanford_cars": ".",
    "sun397": "SUN397",
    "ucf101": "UCF-101-midframes",
}

_CLASS_COUNTS = {
    "caltech101": 100,
    "dtd": 47,
    "eurosat": 10,
    "fgvc": 100,
    "food101": 101,
    "oxford_flowers": 102,
    "oxford_pets": 37,
    "stanford_cars": 196,
    "sun397": 397,
    "ucf101": 101,
}

_SPLIT_FILES = {
    "caltech101": "split_zhou_Caltech101.json",
    "dtd": "split_zhou_DescribableTextures.json",
    "eurosat": "split_zhou_EuroSAT.json",
    "food101": "split_zhou_Food101.json",
    "oxford_flowers": "split_zhou_OxfordFlowers.json",
    "oxford_pets": "split_zhou_OxfordPets.json",
    "stanford_cars": "split_zhou_StanfordCars.json",
    "sun397": "split_zhou_SUN397.json",
    "ucf101": "split_zhou_UCF101.json",
}

_DATA_DIRECTORIES = {
    name: (image_dir,)
    for name, image_dir in _IMAGE_DIRS.items()
    if image_dir != "."
}
_DATA_DIRECTORIES["stanford_cars"] = ("cars_train", "cars_test")

_DATA_FILES = {
    "fgvc": (
        "variants.txt",
        "images_variant_train.txt",
        "images_variant_val.txt",
        "images_variant_test.txt",
    ),
}

_SPLIT_CATALOG_PATH = Path(__file__).with_name("split_catalog.json")
with _SPLIT_CATALOG_PATH.open(encoding="utf-8") as _catalog_file:
    _SPLIT_CATALOG = json.load(_catalog_file)


class Official2SFSDataError(RuntimeError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path, expected, description):
    actual = _sha256(path)
    if actual != expected:
        raise Official2SFSDataError(
            f"{description} checksum mismatch: {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )


def _load_jsonl(path):
    try:
        with path.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise Official2SFSDataError(
            f"Could not read the official 2SFS manifest: {path}"
        ) from error


def _safe_relative_path(base, value, source):
    if not isinstance(value, str) or not value:
        raise Official2SFSDataError(f"Invalid image path in {source}: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise Official2SFSDataError(
            f"Image path must be relative and stay inside its dataset: {source}"
        )
    return (base / relative).resolve()


def _read_json_split(dataset_root, dataset_name):
    split_path = dataset_root / _SPLIT_FILES[dataset_name]
    _verify_checksum(
        split_path,
        _SPLIT_CATALOG["source_splits"][dataset_name]["sha256"],
        "Official CoOp split",
    )
    try:
        with split_path.open(encoding="utf-8") as file:
            raw_split = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise Official2SFSDataError(
            f"Could not read the official CoOp split: {split_path}"
        ) from error

    if not isinstance(raw_split, dict) or set(raw_split) != {"train", "val", "test"}:
        raise Official2SFSDataError(
            f"Official CoOp split must contain train, val, and test: {split_path}"
        )

    image_root = dataset_root / _IMAGE_DIRS[dataset_name]
    partitions = {}
    for split_name in ("train", "val", "test"):
        records = {}
        for raw_item in raw_split[split_name]:
            if not isinstance(raw_item, list) or len(raw_item) != 3:
                raise Official2SFSDataError(
                    f"Invalid {split_name} record in CoOp split: {split_path}"
                )
            impath, label, classname = raw_item
            if (
                type(label) is not int
                or not isinstance(classname, str)
                or not classname
            ):
                raise Official2SFSDataError(
                    f"Invalid {split_name} label/classname in: {split_path}"
                )
            resolved = _safe_relative_path(image_root, impath, split_path)
            if resolved in records:
                raise Official2SFSDataError(
                    f"Duplicate {split_name} image in CoOp split: {split_path}"
                )
            records[resolved] = (label, classname)
        partitions[split_name] = records
    return partitions, [split_path]


def _read_fgvc_split(dataset_root):
    variants_path = dataset_root / "variants.txt"
    expected_checksums = _SPLIT_CATALOG["source_splits"]["fgvc"]["sha256"]
    _verify_checksum(
        variants_path,
        expected_checksums[variants_path.name],
        "Official FGVC annotation",
    )
    try:
        classnames = variants_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise Official2SFSDataError(
            f"Could not read the official FGVC classes: {variants_path}"
        ) from error
    if len(classnames) != len(set(classnames)) or any(not name for name in classnames):
        raise Official2SFSDataError(f"Invalid FGVC classes: {variants_path}")
    classname_to_label = {name: label for label, name in enumerate(classnames)}

    source_paths = [variants_path]
    partitions = {}
    for split_name in ("train", "val", "test"):
        split_path = dataset_root / f"images_variant_{split_name}.txt"
        source_paths.append(split_path)
        _verify_checksum(
            split_path,
            expected_checksums[split_path.name],
            "Official FGVC annotation",
        )
        try:
            lines = split_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise Official2SFSDataError(
                f"Could not read the official FGVC split: {split_path}"
            ) from error
        records = {}
        for line in lines:
            image_id, separator, classname = line.partition(" ")
            if not separator or classname not in classname_to_label:
                raise Official2SFSDataError(
                    f"Invalid FGVC {split_name} record in: {split_path}"
                )
            resolved = _safe_relative_path(
                dataset_root / "images", f"{image_id}.jpg", split_path
            )
            if resolved in records:
                raise Official2SFSDataError(
                    f"Duplicate FGVC {split_name} image in: {split_path}"
                )
            records[resolved] = (classname_to_label[classname], classname)
        partitions[split_name] = records
    return partitions, source_paths


def _load_source_partitions(dataset_root, dataset_name, expected_classes):
    if dataset_name == "fgvc":
        partitions, source_paths = _read_fgvc_split(dataset_root)
    else:
        partitions, source_paths = _read_json_split(dataset_root, dataset_name)

    label_to_classname = {}
    path_to_partition = {}
    missing_images = []
    for split_name, records in partitions.items():
        labels = {label for label, _ in records.values()}
        if labels != set(range(expected_classes)):
            raise Official2SFSDataError(
                f"Official {dataset_name} {split_name} split must contain labels "
                f"0..{expected_classes - 1}"
            )
        for image_path, (label, classname) in records.items():
            if not image_path.is_file() and len(missing_images) < 10:
                missing_images.append(image_path)
            previous_partition = path_to_partition.setdefault(image_path, split_name)
            if previous_partition != split_name:
                raise Official2SFSDataError(
                    f"Official {dataset_name} full splits overlap at: {image_path}"
                )
            previous_classname = label_to_classname.setdefault(label, classname)
            if previous_classname != classname:
                raise Official2SFSDataError(
                    f"Label {label} maps to multiple classnames in {dataset_name}"
                )
    if len(set(label_to_classname.values())) != expected_classes:
        raise Official2SFSDataError(
            f"Classnames are not one-to-one in the official {dataset_name} split"
        )
    if missing_images:
        formatted = "\n".join(f"  - {path}" for path in missing_images)
        raise Official2SFSDataError(
            f"Official {dataset_name} split references missing images "
            f"(showing up to 10):\n{formatted}"
        )
    return partitions, source_paths


def _validate_manifest(
    path,
    expected_per_class,
    expected_classes,
    expected_sha256,
    root,
    source_records,
):
    _verify_checksum(path, expected_sha256, "Official 2SFS manifest")

    items = _load_jsonl(path)
    required = {"_impath", "_label", "_classname", "_domain"}
    if any(not isinstance(item, dict) or set(item) != required for item in items):
        raise Official2SFSDataError(
            f"Official 2SFS manifest has an invalid record: {path}"
        )

    for item in items:
        if (
            not isinstance(item["_impath"], str)
            or not item["_impath"]
            or type(item["_label"]) is not int
            or not isinstance(item["_classname"], str)
            or not item["_classname"]
            or type(item["_domain"]) is not int
            or item["_domain"] != 0
        ):
            raise Official2SFSDataError(
                f"Official 2SFS manifest has invalid field types: {path}"
            )

    counts = Counter(item["_label"] for item in items)
    expected_labels = set(range(expected_classes))
    if set(counts) != expected_labels or set(counts.values()) != {expected_per_class}:
        raise Official2SFSDataError(
            f"Official 2SFS manifest must contain {expected_per_class} unique "
            f"samples for each of {expected_classes} classes: {path}"
        )

    resolved_paths = []
    label_to_classname = {}
    classname_to_label = {}
    for item in items:
        resolved = _safe_relative_path(root, item["_impath"], path)
        resolved_paths.append(resolved)
        expected = source_records.get(resolved)
        actual = (item["_label"], item["_classname"])
        if expected != actual:
            raise Official2SFSDataError(
                f"Few-shot record is not in its official CoOp partition: "
                f"{item['_impath']}"
            )
        if not resolved.is_file():
            raise Official2SFSDataError(
                f"Official 2SFS image is missing: {resolved}"
            )
        previous_name = label_to_classname.setdefault(*actual)
        previous_label = classname_to_label.setdefault(actual[1], actual[0])
        if previous_name != actual[1] or previous_label != actual[0]:
            raise Official2SFSDataError(
                f"Labels and classnames are not one-to-one in: {path}"
            )

    if len(resolved_paths) != len(set(resolved_paths)):
        raise Official2SFSDataError(
            f"Official 2SFS manifest contains duplicate image paths: {path}"
        )
    return items


def _official_paths(root, dataset_name, shots, split_seed):
    dataset_root = root / _DATASET_DIRS[dataset_name]
    split_dir = dataset_root / "split_fewshot"
    train_manifest = split_dir / f"shot_{shots}-seed_{split_seed}_train.jsonl"
    val_manifest = split_dir / f"shot_{shots}-seed_{split_seed}_val.jsonl"

    required_directories = [
        dataset_root,
        *(dataset_root / marker for marker in _DATA_DIRECTORIES[dataset_name]),
    ]
    required_files = [
        *(dataset_root / marker for marker in _DATA_FILES.get(dataset_name, ())),
        train_manifest,
        val_manifest,
    ]
    if dataset_name in _SPLIT_FILES:
        required_files.append(dataset_root / _SPLIT_FILES[dataset_name])

    missing = [path for path in required_directories if not path.is_dir()]
    missing.extend(path for path in required_files if not path.is_file())
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise Official2SFSDataError(
            "Official 2SFS data is incomplete. Follow the CoOp dataset layout "
            "and download the public 2SFS JSONL splits. Missing paths:\n"
            f"{formatted}"
        )
    return dataset_root, train_manifest, val_manifest


def _protocol_metadata(
    dataset_name,
    shots,
    setting,
    split_seed,
    training_seed,
    train_manifest,
    val_manifest,
    source_paths,
    template,
):
    return {
        "name": "official-2sfs",
        "upstream_commit": UPSTREAM_COMMIT,
        "split_catalog": _SPLIT_CATALOG["schema_version"],
        "dataset": dataset_name,
        "shots": shots,
        "setting": setting,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "augmentation_seed_policy": "per_visit_from_training_seed",
        "prompt_template": template,
        "train_transform": (
            "RandomResizedCrop(224,scale=(0.08,1.0),bicubic);"
            "RandomHorizontalFlip(0.5);CLIPNormalize"
        ),
        "test_transform": "Resize(224,bicubic);CenterCrop(224);CLIPNormalize",
        "validation_scope": "base_only" if setting == "base2new" else "all",
        "validation_usage": "loaded_not_used_for_training_or_model_selection",
        "train_manifest_sha256": _sha256(train_manifest),
        "validation_manifest_sha256": _sha256(val_manifest),
        "source_split_sha256": {
            path.name: _sha256(path) for path in source_paths
        },
    }


class OfficialDatumDataset(Dataset):
    def __init__(
        self, data_source, root, transform, classnames, template, protocol
    ):
        self.data_source = list(data_source)
        self.root = Path(root)
        self.transform = transform
        self.classes = [(classname,) for classname in classnames]
        self.template = template
        self.targets = [item.label for item in self.data_source]
        self.impaths = [item.impath for item in self.data_source]
        self.protocol = protocol

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        transform_seed = None
        if isinstance(index, tuple):
            index, transform_seed = index
        item = self.data_source[index]
        image_path = Path(item.impath)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        try:
            with Image.open(image_path) as file:
                image = file.convert("RGB")
        except OSError as error:
            raise Official2SFSDataError(
                f"Could not read official 2SFS image: {image_path}"
            ) from error
        if self.transform is not None and transform_seed is None:
            image = self.transform(image)
        elif self.transform is not None:
            python_state = random.getstate()
            try:
                random.seed(transform_seed)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(transform_seed)
                    image = self.transform(image)
            finally:
                random.setstate(python_state)
        return image, item.label


class SeededRandomSampler(Sampler):
    """Yield an index and a deterministic, per-visit augmentation seed."""

    def __init__(self, data_source, generator):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        size = len(self.data_source)
        indices = torch.randperm(size, generator=self.generator).tolist()
        seeds = torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (size,),
            generator=self.generator,
        ).tolist()
        return iter(zip(indices, seeds))

    def __len__(self):
        return len(self.data_source)


def build_official_2sfs_loaders(
    batch_size,
    dataset_name,
    root="data",
    shots=16,
    setting="base2new",
    split_seed=1,
    training_seed=2026,
    test_batch_size=None,
    num_workers=8,
):
    if dataset_name not in OFFICIAL_2SFS_DATASETS:
        choices = ", ".join(OFFICIAL_2SFS_DATASETS)
        raise ValueError(
            f"Unsupported official 2SFS dataset: {dataset_name}. Choose from {choices}"
        )
    if shots not in (1, 2, 4, 8, 16):
        raise ValueError("Official 2SFS shots must be one of 1, 2, 4, 8, or 16")
    if split_seed not in OFFICIAL_SPLIT_SEEDS:
        raise ValueError("Official 2SFS split_seed must be 1, 2, or 3")
    if setting not in ("standard", "base2new"):
        raise ValueError("Official 2SFS setting must be standard or base2new")
    if num_workers < 0:
        raise ValueError("Official 2SFS num_workers cannot be negative")

    root = Path(root).expanduser().resolve()
    dataset_root, train_manifest, val_manifest = _official_paths(
        root, dataset_name, shots, split_seed
    )
    source_partitions, source_paths = _load_source_partitions(
        dataset_root, dataset_name, _CLASS_COUNTS[dataset_name]
    )
    manifest_key = f"{dataset_name}/{shots}/{split_seed}"
    train_items = _validate_manifest(
        train_manifest,
        shots,
        _CLASS_COUNTS[dataset_name],
        _SPLIT_CATALOG["manifests"][f"{manifest_key}/train"],
        root,
        source_partitions["train"],
    )
    val_items = _validate_manifest(
        val_manifest,
        min(shots, 4),
        _CLASS_COUNTS[dataset_name],
        _SPLIT_CATALOG["manifests"][f"{manifest_key}/val"],
        root,
        source_partitions["val"],
    )
    train_paths = {item["_impath"] for item in train_items}
    val_paths = {item["_impath"] for item in val_items}
    if train_paths & val_paths:
        raise Official2SFSDataError(
            "Official 2SFS train and validation manifests overlap"
        )

    python_random_state = random.getstate()
    try:
        dataset = build_dataset(
            dataset=dataset_name,
            root_path=str(root),
            shots=shots,
            setting=setting,
            seed=split_seed,
        )
    finally:
        random.setstate(python_random_state)

    train_transform, test_transform = build_transforms(224, clip=True)
    template = dataset.template[0]
    protocol = _protocol_metadata(
        dataset_name,
        shots,
        setting,
        split_seed,
        training_seed,
        train_manifest,
        val_manifest,
        source_paths,
        template,
    )

    train_dataset = OfficialDatumDataset(
        dataset.train_x,
        root,
        train_transform,
        dataset.classnames,
        template,
        protocol,
    )
    validation_dataset = OfficialDatumDataset(
        dataset.val,
        root,
        test_transform,
        dataset.val_classnames,
        template,
        protocol,
    )
    test_dataset = OfficialDatumDataset(
        dataset.test,
        root,
        test_transform,
        dataset.test_classnames,
        template,
        protocol,
    )

    sampler_generator = torch.Generator().manual_seed(training_seed)
    worker_generator = torch.Generator().manual_seed(training_seed)
    train_sampler = SeededRandomSampler(train_dataset, sampler_generator)
    pin_memory = torch.cuda.is_available()
    test_batch_size = test_batch_size or batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        generator=worker_generator,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    if dataset.test_new is not None:
        test_new_dataset = OfficialDatumDataset(
            dataset.test_new,
            root,
            test_transform,
            dataset.test_new_classnames,
            template,
            protocol,
        )
        test_new_loader = DataLoader(
            test_new_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=pin_memory,
        )
        test_loader = (test_loader, test_new_loader)

    return train_loader, validation_loader, test_loader, len(train_dataset.classes)
