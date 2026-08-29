"""Training, evaluation, and artifacts for the Kaggle 2SFS notebooks."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.architecture import CLIP_MODEL, load_clip
from src.methods.twostage import TwoStageCLIP
from src.peft import (
    apply_lora,
    mark_only_layernorm_as_trainable,
    mark_only_lora_as_trainable,
)

from .data import (
    DATASETS,
    EXPECTED_TEST_COUNTS,
    ManifestDataset,
    clip_transforms,
    load_train_dataset,
    load_validation_datasets,
    validation_manifest_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "results/logs/kaggle_breakpoint"
FIXED_RATIOS = {"ln": 0.6, "lora": 0.3}
FINAL_EVALUATION_PROTOCOL = "official_test_only_v1"
SUBMISSION_PREDICTION_SCHEMA = "id_class_key_v1"


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    datasets: tuple[str, ...] = DATASETS
    shots: int = 16
    peft: str = "ln"
    batch_size: int = 32
    lr: float = 2e-4
    steps_per_shot: int = 300
    probe_every_steps: int = 10
    seed: int = 2026
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    print_every_steps: int = 100
    model_name: str = CLIP_MODEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())
        object.__setattr__(self, "datasets", tuple(self.datasets))
        if not self.repo_root.is_dir():
            raise FileNotFoundError(self.repo_root)
        if not self.datasets or set(self.datasets) - set(DATASETS):
            raise ValueError(f"Unsupported datasets: {self.datasets}")
        if self.shots not in {1, 2, 4, 8, 16} or self.peft not in FIXED_RATIOS:
            raise ValueError("Invalid shots or PEFT method")
        if (
            min(self.batch_size, self.lr, self.steps_per_shot, self.probe_every_steps)
            <= 0
        ):
            raise ValueError("Training settings must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")

    @property
    def total_steps(self) -> int:
        return self.shots * self.steps_per_shot

    @property
    def fixed_ratio(self) -> float:
        return FIXED_RATIOS[self.peft]

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repo_root"] = str(self.repo_root)
        payload["datasets"] = list(self.datasets)
        payload["fixed_stage_one_ratio"] = self.fixed_ratio
        payload["total_steps"] = self.total_steps
        return payload


@dataclass
class BreakpointRun:
    method: TwoStageCLIP
    parameters: list[torch.nn.Parameter]
    train_loader: DataLoader
    validation: tuple[ManifestDataset, ManifestDataset]
    device: torch.device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(config: ExperimentConfig) -> torch.device:
    requested = config.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_loader(
    dataset: Dataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=resolve_device(config).type == "cuda",
        generator=torch.Generator().manual_seed(config.seed + seed_offset),
        persistent_workers=config.num_workers > 0,
    )


def prepare_method(
    config: ExperimentConfig,
    dataset: ManifestDataset,
    device: torch.device,
) -> tuple[TwoStageCLIP, list[torch.nn.Parameter]]:
    model, tokenizer = load_clip(config.model_name)
    method = TwoStageCLIP(model, tokenizer, dataset.classes, dataset.template).to(
        device
    )
    if config.peft == "lora":
        apply_lora(method.model.vision_model)
        apply_lora(method.model.text_model)
        parameters = mark_only_lora_as_trainable(method.model)
    else:
        parameters = mark_only_layernorm_as_trainable(method.model)
    if not parameters:
        raise RuntimeError(f"No trainable {config.peft} parameters were found")
    return method, list(parameters)


def prepare_breakpoint(config: ExperimentConfig, dataset: str) -> BreakpointRun:
    """Prepare Stage 1 without reading test data or labels."""

    seed_everything(config.seed)
    device = resolve_device(config)
    train_transform, eval_transform = clip_transforms()
    train_data = load_train_dataset(config, dataset, train_transform)
    validation = load_validation_datasets(config, dataset, eval_transform)
    method, parameters = prepare_method(config, train_data, device)
    loader = make_loader(train_data, config, shuffle=True, seed_offset=101)
    return BreakpointRun(method, parameters, loader, validation, device)


def _metrics(correct: Mapping[str, int], total: Mapping[str, int]) -> dict[str, Any]:
    if any(total.get(split, 0) == 0 for split in ("base", "novel")):
        raise ValueError("Validation/evaluation split is empty")
    base = correct.get("base", 0) / total["base"]
    novel = correct.get("novel", 0) / total["novel"]
    harmonic = 0.0 if base + novel == 0 else 2 * base * novel / (base + novel)
    return {
        "base_accuracy": float(base),
        "novel_accuracy": float(novel),
        "harmonic_mean": float(harmonic),
        "base_total": int(total["base"]),
        "novel_total": int(total["novel"]),
    }


@torch.inference_mode()
def evaluate(
    method: TwoStageCLIP,
    datasets: tuple[ManifestDataset, ManifestDataset],
    classifiers: tuple[torch.Tensor, torch.Tensor],
    config: ExperimentConfig,
    device: torch.device,
    collect_predictions: bool = False,
) -> dict[str, Any]:
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    prediction_keys: dict[str, list[str]] = {}
    method.eval()
    for split, dataset, classifier in zip(("base", "novel"), datasets, classifiers):
        indices: list[int] = []
        for images, labels in make_loader(
            dataset, config, shuffle=False, seed_offset=701
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = method.classifier_logits(images, classifier).argmax(dim=1)
            correct[split] += int(predictions.eq(labels).sum().item())
            total[split] += int(labels.numel())
            if collect_predictions:
                indices.extend(predictions.cpu().tolist())
        if collect_predictions:
            prediction_keys[split] = [dataset.catalog[index].key for index in indices]
    metrics = _metrics(correct, total)
    if collect_predictions:
        metrics["prediction_keys_by_split"] = prediction_keys
    return metrics


def train_steps(
    logits_fn: Any,
    parameters: Sequence[torch.nn.Parameter],
    loader: DataLoader,
    steps: int,
    config: ExperimentConfig,
    device: torch.device,
    name: str,
    probe: Callable[[int], None] | None = None,
) -> None:
    if steps <= 0:
        return
    parameters = list(parameters)
    if not parameters:
        raise RuntimeError(f"{name}: no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=1e-6
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    step = 0
    while step < steps:
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = F.cross_entropy(logits_fn(images), labels)
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                continue
            scheduler.step()
            step += 1
            if step == 1 or step == steps or step % config.print_every_steps == 0:
                print(f"{name} [{step}/{steps}] loss={loss.item():.5f}")
            if probe and (step % config.probe_every_steps == 0 or step == steps):
                probe(step)
            if step >= steps:
                break


# Artifacts


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def save_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)
    return target


def artifact_path(dataset: str, peft: str, shots: int, kind: str) -> Path:
    return ARTIFACT_DIR / f"{kind}_{dataset}_{peft}_{shots}shot.json"


def breakpoint_signature(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "datasets": list(config.datasets),
        "shots": config.shots,
        "peft": config.peft,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "steps_per_shot": config.steps_per_shot,
        "probe_every_steps": config.probe_every_steps,
        "validation_protocol": "complete_raw_official_val_v1",
        "seed": config.seed,
        "amp": config.amp,
        "num_workers": config.num_workers,
        "device": config.device,
        "resolved_device": str(resolve_device(config)),
        "model_name": config.model_name,
    }


def open_breakpoint_checkpoint(
    config: ExperimentConfig, dataset: str, restart: bool
) -> tuple[Path, dict[str, Any]]:
    path = artifact_path(dataset, config.peft, config.shots, "breakpoints")
    signature = breakpoint_signature(config)
    if path.is_file() and not restart:
        payload = load_json(path)
        if payload.get("signature") != signature:
            raise ValueError(f"Different settings already exist in {path}")
    else:
        payload = {
            "kind": "kaggle_2sfs_breakpoints",
            "signature": signature,
            "config": config.as_json(),
            "test_labels_read": False,
            "datasets": {},
        }
    return path, payload


def build_breakpoint_result(
    config: ExperimentConfig,
    dataset: str,
    validation: tuple[ManifestDataset, ManifestDataset],
    records: Sequence[Mapping[str, float | int]],
    selected: Mapping[str, float | int],
) -> dict[str, Any]:
    metric_keys = (
        "base_accuracy",
        "novel_accuracy",
        "harmonic_mean",
        "base_total",
        "novel_total",
    )
    return {
        "dataset": dataset,
        "selection_rule": "earliest probe with maximum full official validation H",
        "validation_protocol": "complete raw official validation split",
        "validation_source": str(validation_manifest_path(config, dataset)),
        "test_labels_read": False,
        "zero_shot_full": {key: records[0][key] for key in metric_keys},
        "validation_samples": sum(map(len, validation)),
        "validation_samples_by_split": {
            "base": len(validation[0]),
            "novel": len(validation[1]),
        },
        "probe_records": records,
        **selected,
        "manual_override": None,
        "selected_step": selected["auto_step"],
        "selected_ratio": selected["auto_ratio"],
        "selected_source": "automatic",
    }


def comparison_signature(
    config: ExperimentConfig, found_ratios: Mapping[str, float]
) -> dict[str, Any]:
    return {
        "base": breakpoint_signature(config),
        "evaluation_protocol": FINAL_EVALUATION_PROTOCOL,
        "prediction_schema": SUBMISSION_PREDICTION_SCHEMA,
        "fixed_ratio": config.fixed_ratio,
        "found_ratios": {name: float(found_ratios[name]) for name in config.datasets},
    }


def open_comparison_checkpoint(
    config: ExperimentConfig,
    found_ratios: Mapping[str, float],
    restart: bool,
) -> tuple[Path, dict[str, Any]]:
    path = artifact_path(config.datasets[0], config.peft, config.shots, "comparison")
    signature = comparison_signature(config, found_ratios)
    if path.is_file() and not restart:
        payload = load_json(path)
        if payload.get("signature") != signature:
            raise ValueError(f"Different settings already exist in {path}")
    else:
        payload = {
            "kind": "kaggle_2sfs_fixed_vs_found",
            "signature": signature,
            "config": config.as_json(),
            "evaluation_protocol": FINAL_EVALUATION_PROTOCOL,
            "prediction_schema": SUBMISSION_PREDICTION_SCHEMA,
            "test_labels_usage": "final evaluation only",
            "model_features": "image pixels only; competition IDs are output-only sidecars",
            "runs": {name: {} for name in config.datasets},
            "summary": {},
        }
    return path, payload


def build_comparison_result(
    config: ExperimentConfig,
    dataset: str,
    ratio: float,
    stage_one_steps: int,
    stage_two_steps: int,
    metrics: Mapping[str, Any],
    ids_by_split: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    metrics = dict(metrics)
    predictions = metrics.pop("prediction_keys_by_split")
    submission_rows = [
        {"id": sample_id, "prediction": prediction}
        for split in ("base", "novel")
        for sample_id, prediction in zip(
            ids_by_split[split], predictions[split], strict=True
        )
    ]
    return {
        "dataset": dataset,
        "evaluation_protocol": FINAL_EVALUATION_PROTOCOL,
        "prediction_schema": SUBMISSION_PREDICTION_SCHEMA,
        "ratio": ratio,
        "stage_one_steps": stage_one_steps,
        "stage_two_steps": stage_two_steps,
        "fresh_seed": config.seed,
        "submission_rows": submission_rows,
        **metrics,
    }


def has_complete_predictions(run: Mapping[str, Any], dataset: str) -> bool:
    rows = run.get("submission_rows")
    if run.get("prediction_schema") != SUBMISSION_PREDICTION_SCHEMA:
        return False
    if not isinstance(rows, list) or len(rows) != sum(
        EXPECTED_TEST_COUNTS[dataset].values()
    ):
        return False
    try:
        ids = [row["id"] for row in rows]
        predictions = [row["prediction"] for row in rows]
    except (KeyError, TypeError):
        return False
    return len(set(ids)) == len(rows) and all(ids) and all(predictions)


def comparison_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    fixed_h: list[float] = []
    found_h: list[float] = []
    for dataset, runs in payload["runs"].items():
        if "fixed" not in runs or "found" not in runs:
            continue
        fixed, found = runs["fixed"], runs["found"]
        row = {"fixed_ratio": fixed["ratio"], "found_ratio": found["ratio"]}
        for metric in ("base_accuracy", "novel_accuracy", "harmonic_mean"):
            row[f"fixed_{metric}"] = fixed[metric]
            row[f"found_{metric}"] = found[metric]
            row[f"delta_{metric}"] = found[metric] - fixed[metric]
        rows[dataset] = row
        fixed_h.append(float(fixed["harmonic_mean"]))
        found_h.append(float(found["harmonic_mean"]))
    complete = len(rows) == len(payload["runs"])
    fixed_mean = sum(fixed_h) / len(fixed_h) if complete and fixed_h else None
    found_mean = sum(found_h) / len(found_h) if complete and found_h else None
    return {
        "datasets": rows,
        "complete": complete,
        "fixed_mean_h": fixed_mean,
        "found_mean_h": found_mean,
        "delta_mean_h": None if not complete else found_mean - fixed_mean,
    }


def config_from_breakpoints(
    payload: Mapping[str, Any], dataset: str
) -> ExperimentConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("Breakpoint JSON has no valid config")
    values = dict(raw)
    for key in ("total_steps", "fixed_stage_one_ratio"):
        values.pop(key, None)
    if tuple(values.get("datasets", ())) != (dataset,):
        raise ValueError(f"Breakpoint JSON must contain only {dataset!r}")
    values.update(repo_root=REPO_ROOT, datasets=(dataset,))
    return ExperimentConfig(**values)
