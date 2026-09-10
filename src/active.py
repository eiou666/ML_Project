from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np

from .active_learning import select_batch
from .config import apply_overrides, deep_merge, load_config, ratio_slug
from .data import build_base_datasets, dataset_targets
from .run import RunResult, run_experiment
from .splits import create_explicit_split, create_stratified_split
from .utils import resolve_device


ACTIVE_FIELDS = [
    "seed",
    "strategy",
    "round",
    "label_ratio",
    "labeled_count",
    "validation_count",
    "unlabeled_count",
    "test_accuracy",
    "test_macro_f1",
    "run_dir",
]


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTIVE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _record(
    history_path: Path,
    *,
    seed: int,
    strategy: str,
    round_index: int,
    result: RunResult,
) -> None:
    split = result.data.split
    _append_row(
        history_path,
        {
            "seed": seed,
            "strategy": strategy,
            "round": round_index,
            "label_ratio": split.label_ratio,
            "labeled_count": split.labeled.size,
            "validation_count": split.validation.size,
            "unlabeled_count": split.unlabeled.size,
            "test_accuracy": result.metrics["test"]["accuracy"],
            "test_macro_f1": result.metrics["test"]["macro_f1"],
            "run_dir": str(result.run_dir),
        },
    )


def _history_has(path: Path, strategy: str, round_index: int) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        return any(
            row["strategy"] == strategy and int(row["round"]) == round_index
            for row in csv.DictReader(handle)
        )


def run_active_seed(
    config: dict[str, Any],
    *,
    seed: int,
    strategies: list[str],
    force: bool = False,
) -> Path:
    active_config = config["active"]
    initial_ratio = float(active_config["initial_ratio"])
    query_size = int(active_config["query_size"])
    rounds = int(active_config["rounds"])
    candidate_multiplier = int(active_config["candidate_multiplier"])
    active_output = config["project"].get(
        "active_output_dir", Path(config["project"]["output_dir"]).parent / "active"
    )
    output_root = Path(active_output) / f"seed{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / "active_history.csv"
    if force and history_path.exists():
        history_path.unlink()

    if not force and history_path.exists():
        if all(_history_has(history_path, s, r) for s in strategies for r in range(rounds + 1)):
            print(f"Active learning seed {seed} already fully completed; skipping.", flush=True)
            return history_path

    base_datasets = build_base_datasets(config)
    targets = dataset_targets(base_datasets[0])
    initial_split = create_stratified_split(
        targets,
        initial_ratio,
        seed,
        float(config["data"]["validation_fraction"]),
    )
    initial_run_config = deep_merge(
        config,
        {
            "data": {"label_ratio": initial_ratio},
            "method": {"name": "fixmatch"},
            "train": {"seed": seed},
            "active": {"strategy": "none"},
        },
    )
    initial_result = run_experiment(
        initial_run_config,
        split=initial_split,
        base_datasets=base_datasets,
        run_dir=output_root / "shared" / f"round0_{ratio_slug(initial_ratio)}",
        force=force,
    )

    for strategy in strategies:
        if not _history_has(history_path, strategy, 0):
            _record(
                history_path,
                seed=seed,
                strategy=strategy,
                round_index=0,
                result=initial_result,
            )
        current_labeled = initial_split.labeled.copy()
        fixed_validation = initial_split.validation.copy()
        current_pool = initial_split.unlabeled.copy()
        current_model = initial_result.model
        current_transform = initial_result.data.evaluation_transform
        device = resolve_device(str(config["train"].get("device", "auto")))

        for round_index in range(1, rounds + 1):
            selected = select_batch(
                current_model,
                strategy=strategy,
                base_dataset=base_datasets[0],
                pool_indices=current_pool,
                transform=current_transform,
                device=device,
                query_size=query_size,
                candidate_multiplier=candidate_multiplier,
                seed=seed * 10_000 + round_index * 101,
                num_workers=int(config["data"]["num_workers"]),
            )
            selected_targets = targets[selected]
            selection_path = output_root / strategy / f"round{round_index}_selection.csv"
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            with selection_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["index", "revealed_label"])
                writer.writeheader()
                writer.writerows(
                    {"index": int(index), "revealed_label": int(label)}
                    for index, label in zip(selected, selected_targets)
                )

            current_labeled = np.sort(np.concatenate([current_labeled, selected]))
            current_pool = np.setdiff1d(current_pool, selected, assume_unique=True)
            ratio = float((current_labeled.size + fixed_validation.size) / len(base_datasets[0]))
            split = create_explicit_split(
                labeled=current_labeled,
                validation=fixed_validation,
                universe_size=len(base_datasets[0]),
                seed=seed,
                label_ratio=ratio,
            )
            round_config = deep_merge(
                config,
                {
                    "data": {"label_ratio": ratio},
                    "method": {"name": "fixmatch"},
                    "train": {"seed": seed},
                    "active": {"strategy": strategy},
                },
            )
            result = run_experiment(
                round_config,
                split=split,
                base_datasets=base_datasets,
                run_dir=output_root / strategy / f"round{round_index}_{ratio:.3f}",
                force=force,
            )
            if not _history_has(history_path, strategy, round_index):
                _record(
                    history_path,
                    seed=seed,
                    strategy=strategy,
                    round_index=round_index,
                    result=result,
                )
            current_model = result.model
            current_transform = result.data.evaluation_transform
            gc.collect()
    return history_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run active-learning rounds for one seed")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "entropy", "uncertainty_diversity"],
        choices=["random", "entropy", "uncertainty_diversity"],
    )
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), seed=args.seed, device=args.device)
    history = run_active_seed(
        config,
        seed=args.seed,
        strategies=args.strategies,
        force=args.force,
    )
    print(json.dumps({"active_history": str(history.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
