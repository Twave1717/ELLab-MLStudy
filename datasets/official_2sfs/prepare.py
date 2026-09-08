"""Download and verify the public few-shot manifests released by 2SFS."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import gdown


CATALOG_PATH = Path(__file__).with_name("split_catalog.json")
DATASET_DIRS = {
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
SUPPORTED_DATA_DIRS = set(DATASET_DIRS.values())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(archive):
    for member in archive.getmembers():
        path = PurePosixPath(member.name.removeprefix("./"))
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise RuntimeError(f"Unsafe path in official split archive: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Unsupported archive entry: {member.name}")
        if path.parts and path.parts[0] not in SUPPORTED_DATA_DIRS:
            continue
        yield member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root containing datasets in the canonical CoOp layout",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="official-2sfs-") as temporary:
        temporary_root = Path(temporary)
        for shots in (1, 2, 4, 8, 16):
            entry = catalog["archives"][str(shots)]
            archive_path = temporary_root / entry["filename"]
            print(f"Downloading official {shots}-shot bundle...")
            downloaded = gdown.download(
                id=entry["google_drive_file_id"],
                output=str(archive_path),
                quiet=False,
            )
            if downloaded is None or not archive_path.is_file():
                raise RuntimeError(f"Download failed for {entry['filename']}")
            actual = sha256(archive_path)
            if actual != entry["sha256"]:
                raise RuntimeError(
                    f"Checksum mismatch for {entry['filename']}: "
                    f"expected {entry['sha256']}, got {actual}"
                )
            with tarfile.open(archive_path) as archive:
                archive.extractall(
                    data_root,
                    members=safe_members(archive),
                    filter="data",
                )

        for dataset_name, entry in catalog["source_splits"].items():
            if dataset_name == "fgvc":
                continue
            temporary_path = temporary_root / entry["filename"]
            print(f"Downloading official CoOp split for {dataset_name}...")
            downloaded = gdown.download(
                id=entry["google_drive_file_id"],
                output=str(temporary_path),
                quiet=False,
            )
            if downloaded is None or not temporary_path.is_file():
                raise RuntimeError(f"Download failed for {entry['filename']}")
            actual = sha256(temporary_path)
            if actual != entry["sha256"]:
                raise RuntimeError(
                    f"Checksum mismatch for {entry['filename']}: "
                    f"expected {entry['sha256']}, got {actual}"
                )
            destination_root = data_root / DATASET_DIRS[dataset_name]
            destination_root.mkdir(parents=True, exist_ok=True)
            destination = destination_root / entry["filename"]
            shutil.copyfile(temporary_path, destination)
            destination.chmod(0o644)

    print(f"Installed verified official 2SFS splits under {data_root}")


if __name__ == "__main__":
    main()
