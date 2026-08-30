"""Read experiment artifacts and build Kaggle submissions."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import data, experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_SUBMISSION_FILENAME = "submission_fixed_ratio_0.60.csv"
FOUND_SUBMISSION_FILENAME = "submission_found_breakpoints.csv"


def _load_artifacts(kind: str, peft: str, shots: int) -> list[dict[str, Any]]:
    paths = [
        experiment.artifact_path(dataset, peft, shots, kind)
        for dataset in data.DATASETS
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Complete all three runs first:\n" + "\n".join(f"- {p}" for p in missing)
        )
    return [experiment.load_json(path) for path in paths]


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def text(value: Any) -> str:
        if value is None:
            return "-"
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    rendered = [[text(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered
    )
    return "\n".join(lines)


def input_summary(config: experiment.ExperimentConfig) -> str:
    """Format class, train, and validation counts."""

    rows = data.describe_breakpoint_inputs(config)
    keys = (
        "dataset",
        "classes",
        "base_classes",
        "novel_classes",
        "train_images",
        "validation_images",
    )
    return _table(
        ("dataset", "classes", "B cls", "N cls", "train", "val"),
        [[row[key] for key in keys] for row in rows],
    )


def breakpoint_summary(peft: str = "ln", shots: int = 16) -> str:
    """Format the selected breakpoint and validation score."""

    results = {
        dataset: result
        for payload in _load_artifacts("breakpoints", peft, shots)
        for dataset, result in payload["datasets"].items()
    }
    rows = [
        (
            dataset,
            results[dataset]["validation_samples"],
            results[dataset]["auto_step"],
            results[dataset]["auto_ratio"],
            results[dataset]["auto_harmonic_mean"],
            results[dataset]["selected_ratio"],
            results[dataset]["selected_source"],
        )
        for dataset in data.DATASETS
    ]
    return _table(
        ("dataset", "val", "auto step", "auto ratio", "best H", "selected", "source"),
        rows,
    )


def comparison_summary(peft: str = "ln", shots: int = 16) -> str:
    """Format per-dataset and macro Fixed-vs-Found metrics."""

    merged: dict[str, Any] = {"runs": {}}
    for payload in _load_artifacts("comparison", peft, shots):
        merged["runs"].update(payload["runs"])
    summary = experiment.comparison_summary(merged)
    metrics = (
        ("base_accuracy", "B"),
        ("novel_accuracy", "N"),
        ("harmonic_mean", "H"),
    )
    rows = [
        (
            dataset,
            row["fixed_ratio"],
            row["found_ratio"],
            *(
                row[f"{prefix}_{metric}"]
                for metric, _ in metrics
                for prefix in ("fixed", "found", "delta")
            ),
        )
        for dataset, row in summary["datasets"].items()
    ]
    headers = (
        "dataset",
        "fixed r",
        "found r",
        *(
            f"{prefix} {short}"
            for _, short in metrics
            for prefix in ("fixed", "found", "delta")
        ),
    )
    table = _table(headers, rows)
    if summary["complete"]:
        table += (
            f"\n\nfixed mean-H={summary['fixed_mean_h']:.4f}, "
            f"found mean-H={summary['found_mean_h']:.4f}, "
            f"delta mean-H={summary['delta_mean_h']:+.4f}"
        )
    return table


def _write_submission(path: Path, rows: Sequence[Mapping[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "prediction"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return path


def export_model_submissions(peft: str = "ln", shots: int = 16) -> dict[str, Path]:
    """Merge saved model predictions in canonical public-test order."""

    comparisons = _load_artifacts("comparison", peft, shots)
    if any(
        payload.get("prediction_schema")
        != experiment.SUBMISSION_PREDICTION_SCHEMA
        for payload in comparisons
    ):
        raise ValueError("Comparison artifacts do not contain model predictions")
    runs = {
        name: run for payload in comparisons for name, run in payload["runs"].items()
    }
    found_ratios = {
        name: float(result["selected_ratio"])
        for payload in _load_artifacts("breakpoints", peft, shots)
        for name, result in payload["datasets"].items()
    }

    public = REPO_ROOT / "kaggle/public"
    test_rows = data.read_csv(public / "test.csv")
    class_rows = data.read_csv(public / "classes.csv")
    test_ids = [row["id"] for row in test_rows]
    test_by_id = {row["id"]: row for row in test_rows}
    expected = sum(
        sum(counts.values()) for counts in data.EXPECTED_TEST_COUNTS.values()
    )
    if len(test_ids) != expected or len(test_by_id) != expected:
        raise ValueError("Public test IDs/count are invalid")
    candidates = {
        (name, split): {
            row["class_key"]
            for row in class_rows
            if row["dataset"] == name and row["class_split"] == split
        }
        for name in data.DATASETS
        for split in ("base", "novel")
    }

    prepared: dict[str, list[dict[str, str]]] = {}
    for label in ("fixed", "found"):
        selected = {name: runs[name][label] for name in data.DATASETS}
        for name, run in selected.items():
            if not experiment.has_complete_predictions(run, name):
                raise ValueError(f"Incomplete {name}/{label} predictions")
            ratio = (
                experiment.FIXED_RATIOS[peft]
                if label == "fixed"
                else found_ratios[name]
            )
            if not math.isclose(
                float(run["ratio"]), ratio, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"Unexpected {name}/{label} ratio")
            if any(
                test_by_id.get(row["id"], {}).get("dataset") != name
                for row in run["submission_rows"]
            ):
                raise ValueError(f"Invalid IDs in {name}/{label}")
        rows = [row for run in selected.values() for row in run["submission_rows"]]
        predictions = {row["id"]: row["prediction"] for row in rows}
        if len(predictions) != len(rows) or set(predictions) != set(test_ids):
            raise ValueError(f"Invalid {label} prediction ID coverage")
        for sample_id, prediction in predictions.items():
            row = test_by_id[sample_id]
            if prediction not in candidates[(row["dataset"], row["eval_split"])]:
                raise ValueError(f"Invalid class prediction for {sample_id}")
        prepared[label] = [
            {"id": sample_id, "prediction": predictions[sample_id]}
            for sample_id in test_ids
        ]

    outputs = {
        "fixed": REPO_ROOT / "results" / FIXED_SUBMISSION_FILENAME,
        "found": REPO_ROOT / "results" / FOUND_SUBMISSION_FILENAME,
    }
    return {
        label: _write_submission(path, prepared[label])
        for label, path in outputs.items()
    }
