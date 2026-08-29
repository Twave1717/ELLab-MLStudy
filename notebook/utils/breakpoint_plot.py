"""Reusable paper-style plots for breakpoint trajectory artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, MaxNLocator  # noqa: E402


CURVE_COLOR = "#0072B2"
NOVEL_PEAK_COLOR = "#CC79A7"
HM_PEAK_COLOR = "#E69F00"
FIXED_COLOR = "#777777"
MANUAL_COLOR = "#111111"

DATASET_LABELS = {
    "eurosat": "EuroSAT",
    "fgvc_aircraft": "FGVC Aircraft",
    "dtd": "DTD",
}


def _records(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = result.get("probe_records")
    if not isinstance(records, list) or not records:
        raise ValueError("Breakpoint result has no probe_records")
    required = {"step", "base_accuracy", "novel_accuracy", "harmonic_mean"}
    if any(not required.issubset(record) for record in records):
        raise ValueError(f"Each probe record must contain: {sorted(required)}")
    return sorted(records, key=lambda record: int(record["step"]))


def _peak_step(
    result: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    field: str,
    metric: str,
) -> int:
    if field in result:
        return int(result[field])
    return int(max(records, key=lambda record: float(record[metric]))["step"])


def _style_axis(ax: plt.Axes, total_steps: int) -> None:
    margin = max(total_steps * 0.02, 1.0)
    ax.set_xlim(-margin, total_steps + margin)
    ax.set_xticks((0, total_steps // 2, total_steps))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(
        which="major",
        color="#4C4C4C",
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )
    ax.tick_params(
        which="major",
        direction="in",
        top=True,
        right=True,
        length=6,
        width=1.1,
    )
    ax.tick_params(
        which="minor",
        direction="in",
        top=True,
        right=True,
        length=3,
        width=0.9,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax.set_xlabel("iterations", fontsize=17, labelpad=7)


def _set_accuracy_limits(ax: plt.Axes, values: Sequence[float]) -> None:
    low, high = min(values), max(values)
    margin = max((high - low) * 0.08, 0.5)
    ax.set_ylim(max(0.0, low - margin), min(100.0, high + margin))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot_breakpoint_result(
    result: Mapping[str, Any],
    output_path: str | Path,
    *,
    fixed_ratio: float = 0.6,
    manual_step: int | None = None,
    dpi: int = 180,
) -> Path:
    """Save a Base/Novel trajectory figure from one dataset result."""

    records = _records(result)
    steps = [int(record["step"]) for record in records]
    total_steps = max(steps)
    if total_steps <= 0:
        raise ValueError("The trajectory must contain a positive training step")

    novel_peak = _peak_step(
        result, records, "novel_peak_step", "novel_accuracy"
    )
    hm_peak = _peak_step(result, records, "hm_peak_step", "harmonic_mean")
    fixed_step = int(total_steps * fixed_ratio)
    dataset = str(result.get("dataset", "dataset"))
    dataset_label = DATASET_LABELS.get(dataset, dataset)

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 13,
            "axes.axisbelow": True,
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 6.0))
    figure.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.84,
        bottom=0.25,
        wspace=0.22,
    )
    figure.supylabel("validation accuracy (%)", x=0.018, fontsize=17)

    for ax, split, linestyle in zip(
        axes, ("base", "novel"), ("-", "--"), strict=True
    ):
        values = [float(record[f"{split}_accuracy"]) * 100 for record in records]
        _style_axis(ax, total_steps)
        _set_accuracy_limits(ax, values)
        ax.axvline(
            fixed_step,
            color=FIXED_COLOR,
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            zorder=2,
        )
        ax.axvline(
            hm_peak,
            color=HM_PEAK_COLOR,
            linestyle="-",
            linewidth=6.0,
            alpha=0.30,
            zorder=2,
        )
        ax.axvline(
            novel_peak,
            color=NOVEL_PEAK_COLOR,
            linestyle="-",
            linewidth=4.0,
            alpha=0.45,
            zorder=3,
        )
        if manual_step is not None:
            ax.axvline(
                manual_step,
                color=MANUAL_COLOR,
                linestyle="-.",
                linewidth=2.2,
                alpha=0.9,
                zorder=4,
            )
        ax.plot(
            steps,
            values,
            color=CURVE_COLOR,
            linestyle=linestyle,
            linewidth=2.8,
            zorder=5,
        )
        ax.set_title(split, fontsize=22, fontstyle="italic", pad=10)

    legend = [
        Line2D(
            [0], [0], color=NOVEL_PEAK_COLOR, linewidth=5.0, alpha=0.55,
            label=f"Novel peak ({novel_peak})"
        ),
        Line2D(
            [0], [0], color=HM_PEAK_COLOR, linewidth=6.0, alpha=0.40,
            label=f"HM max ({hm_peak})"
        ),
        Line2D(
            [0], [0], color=FIXED_COLOR, linestyle=":", linewidth=2.0,
            label=f"fixed {fixed_ratio:.2f} ({fixed_step})"
        ),
    ]
    if manual_step is not None:
        legend.append(
            Line2D(
                [0], [0], color=MANUAL_COLOR, linestyle="-.", linewidth=2.2,
                label=f"manual ({manual_step})"
            )
        )
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=len(legend),
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.035,
        dataset_label,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, facecolor="white")
    plt.close(figure)
    return target


def plot_breakpoint_artifact(
    artifact_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    fixed_ratio: float | None = None,
    manual_step: int | None = None,
    dpi: int = 180,
) -> list[Path]:
    """Regenerate figures from a saved breakpoint JSON artifact."""

    source = Path(artifact_path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"No datasets found in {source}")

    if fixed_ratio is None:
        fixed_ratio = float(
            payload.get("config", {}).get("fixed_stage_one_ratio", 0.6)
        )
    directory = source.parent if output_dir is None else Path(output_dir)
    multiple = len(datasets) > 1
    outputs = []
    for dataset, result in datasets.items():
        name = f"{source.stem}_{dataset}.png" if multiple else f"{source.stem}.png"
        outputs.append(
            plot_breakpoint_result(
                result,
                directory / name,
                fixed_ratio=fixed_ratio,
                manual_step=manual_step,
                dpi=dpi,
            )
        )
    return outputs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate paper-style breakpoint figures from JSON."
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fixed-ratio", type=float)
    parser.add_argument("--manual-step", type=int)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)

    for artifact in args.artifacts:
        outputs = plot_breakpoint_artifact(
            artifact,
            args.output_dir,
            fixed_ratio=args.fixed_ratio,
            manual_step=args.manual_step,
            dpi=args.dpi,
        )
        for output in outputs:
            print(output)


if __name__ == "__main__":
    main()
