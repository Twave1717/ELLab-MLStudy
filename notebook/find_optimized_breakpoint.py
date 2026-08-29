"""Find an offline optimized 2SFS breakpoint on the exact Kaggle split.

The selected point is the earliest Novel accuracy peak on the complete
official Base/Novel validation split.  Every probe is retained for manual
review.  This is a research-time validation
oracle, not a train-only stopping rule.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = next(
    path
    for path in (
        Path.cwd(),
        *Path.cwd().parents,
        *Path(__file__).resolve().parents,
    )
    if (path / "pyproject.toml").is_file()
)
sys.path[:0] = [str(ROOT), str(ROOT / "notebook")]

from train_2sfs import build_parser as build_train_parser  # noqa: E402
from utils import breakpoint_plot, data, experiment  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "archive/05_optimized_breakpoint_experiments"
FINDER_PROTOCOL = "kaggle_full_validation_novel_peak_trajectory_v2"


def build_parser():
    parser = build_train_parser(
        description=(
            "Search the exact Kaggle Base/Novel validation trajectory and "
            "select the earliest Novel accuracy peak."
        )
    )
    # The Kaggle breakpoint protocol is the completed 16-shot setup.  Override
    # train_2sfs.py's generic 1-shot CLI default explicitly.
    parser.set_defaults(dataset="all", setting="base2new", shots=16)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--probe_every_steps", type=int, default=10)
    parser.add_argument("--print_every_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Replace an existing artifact with the same filename.",
    )
    return parser


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def selected_datasets(name: str) -> tuple[str, ...]:
    if name == "all":
        return data.DATASETS
    if name not in data.DATASETS:
        supported = ", ".join((*data.DATASETS, "all"))
        raise ValueError(f"Kaggle breakpoint dataset must be one of: {supported}")
    return (name,)


def artifact_path(output_dir: Path, config, dataset: str) -> Path:
    gate = config.gradient_gate
    return output_dir / (
        f"optimized_breakpoint_{dataset}_{config.peft}_{gate}_"
        f"{config.shots}shot.json"
    )


def save_figure(payload: dict, dataset: str, path: Path, config) -> Path:
    result = payload["datasets"][dataset]
    figure_path = path.with_suffix(".png")
    breakpoint_plot.plot_breakpoint_result(
        result,
        figure_path,
        fixed_ratio=config.fixed_ratio,
    )
    print(f"{dataset}: saved {display_path(figure_path)}")
    return figure_path


def run_dataset(args, dataset: str, output_dir: Path) -> dict:
    config = experiment.ExperimentConfig(
        repo_root=ROOT,
        data_root=resolve_from_root(args.data_root),
        kaggle_root=resolve_from_root(args.kaggle_root),
        datasets=(dataset,),
        shots=args.shots,
        peft=args.peft,
        gradient_gate=args.gradient_gate,
        batch_size=args.batch_size,
        lr=args.lr,
        steps_per_shot=args.steps_per_shot,
        probe_every_steps=args.probe_every_steps,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
        num_workers=args.num_workers,
        print_every_steps=args.print_every_steps,
    )
    path = artifact_path(output_dir, config, dataset)
    signature = {
        **experiment.breakpoint_signature(config),
        "finder_protocol": FINDER_PROTOCOL,
    }

    if path.is_file() and not args.restart:
        payload = experiment.load_json(path)
        if payload.get("signature") != signature:
            raise ValueError(
                f"Different settings already exist in {path}; "
                "use --restart or a different --output_dir."
            )
        print(f"{dataset}: reuse {display_path(path)}")
        save_figure(payload, dataset, path, config)
        return payload

    result = experiment.find_optimized_breakpoint(config, dataset)
    payload = {
        "kind": "kaggle_2sfs_optimized_breakpoint",
        "finder_protocol": FINDER_PROTOCOL,
        "signature": signature,
        "config": config.as_json(),
        "test_labels_read": False,
        "figures": {dataset: path.with_suffix(".png").name},
        "datasets": {dataset: result},
    }
    experiment.save_json(path, payload)
    print(f"{dataset}: saved {display_path(path)}")
    save_figure(payload, dataset, path, config)
    return payload


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.setting != "base2new":
        parser.error("find_optimized_breakpoint requires --setting base2new")
    if args.probe_every_steps <= 0 or args.print_every_steps <= 0:
        parser.error("probe/print intervals must be positive")
    if args.bp_mode is not None or args.stage_one_ratio is not None:
        print(
            "Note: --bp_mode/--stage_one_ratio do not limit the search. "
            "The finder always scans the complete Stage-1 budget."
        )
    if args.gradient_gate == "abs_identity" and args.amp:
        print(
            "Warning: Dynamic Gate AMP overflow handling is not transactional yet. "
            "Use --no-amp for the correctness-first experiment."
        )

    try:
        datasets = selected_datasets(args.dataset)
    except ValueError as error:
        parser.error(str(error))
    output_dir = resolve_from_root(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = {
        dataset: run_dataset(args, dataset, output_dir) for dataset in datasets
    }
    print("\nSelected offline validation Novel peaks")
    for dataset, payload in payloads.items():
        result = payload["datasets"][dataset]
        print(
            f"- {dataset}: step={result['novel_peak_step']}, "
            f"ratio={result['novel_peak_ratio']:.6f}, "
            f"N={result['novel_peak_value']:.4f}; "
            f"HM max step={result['hm_peak_step']}"
        )


if __name__ == "__main__":
    main()
