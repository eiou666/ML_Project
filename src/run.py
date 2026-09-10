from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    apply_overrides,
    config_hash,
    dump_yaml,
    load_config,
    make_run_name,
)
from .data import DataBundle, build_base_datasets, build_data_bundle, dataset_targets
from .engine import load_ema_model_from_checkpoint, train
from .metrics import evaluate
from .models.fashion_cnn import parameter_count
from .splits import SplitIndices, class_count_map, create_stratified_split
from .utils import atomic_json_dump, package_versions, resolve_device, seed_everything


@dataclass
class RunResult:
    metrics: dict[str, Any]
    model: torch.nn.Module
    data: DataBundle
    run_dir: Path


def _combined_hash(config: dict[str, Any], split_hash: str) -> str:
    value = f"{config_hash(config)}:{split_hash}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _latest_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    row = rows[-1]
    numeric_keys = {
        "step",
        "learning_rate",
        "supervised_loss",
        "unsupervised_loss",
        "train_accuracy",
        "pseudo_coverage",
        "pseudo_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "elapsed_seconds",
    }
    return {
        key: float(value) if key in numeric_keys and value != "" else value
        for key, value in row.items()
    }


def _clear_run_outputs(run_dir: Path) -> None:
    for name in [
        "checkpoint_latest.pt",
        "history.csv",
        "metrics.json",
        "test_predictions.npz",
        "split_indices.npz",
        "split_summary.json",
        "resolved_config.yaml",
    ]:
        path = run_dir / name
        if path.exists():
            path.unlink()


def run_experiment(
    config: dict[str, Any],
    *,
    split: SplitIndices | None = None,
    base_datasets: tuple[Any, Any] | None = None,
    run_dir: str | Path | None = None,
    resume: bool = True,
    force: bool = False,
) -> RunResult:
    seed = int(config["train"]["seed"])
    seed_everything(seed, bool(config["train"].get("deterministic", True)))
    device = resolve_device(str(config["train"].get("device", "auto")))
    bases = base_datasets or build_base_datasets(config)
    targets = dataset_targets(bases[0])
    if split is None:
        split = create_stratified_split(
            targets,
            float(config["data"]["label_ratio"]),
            seed,
            float(config["data"]["validation_fraction"]),
        )
    data = build_data_bundle(config, split=split, base_datasets=bases)
    destination = (
        Path(run_dir)
        if run_dir is not None
        else Path(config["project"]["output_dir"]) / make_run_name(config, split_hash=split.split_hash)
    )
    destination.mkdir(parents=True, exist_ok=True)
    if force:
        _clear_run_outputs(destination)

    metrics_path = destination / "metrics.json"
    checkpoint_path = destination / "checkpoint_latest.pt"
    resolved_hash = _combined_hash(config, split.split_hash)
    if metrics_path.exists() and checkpoint_path.exists() and not force:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("resolved_hash") != resolved_hash:
            raise RuntimeError("Completed run configuration or split differs; use a separate output directory or explicitly --force a rerun")
        if metrics.get("train_steps") != int(config["train"]["steps"]):
            raise RuntimeError("Completed run has an unexpected training step count")
        model = load_ema_model_from_checkpoint(config, checkpoint_path, device)
        return RunResult(metrics=metrics, model=model, data=data, run_dir=destination)

    dump_yaml(config, destination / "resolved_config.yaml")
    np.savez_compressed(
        destination / "split_indices.npz",
        labeled=split.labeled,
        validation=split.validation,
        unlabeled=split.unlabeled,
        annotated=split.annotated,
    )
    split_summary = {
        "seed": split.seed,
        "label_ratio": split.label_ratio,
        "split_hash": split.split_hash,
        "counts": {
            "labeled": int(split.labeled.size),
            "validation": int(split.validation.size),
            "annotated": int(split.annotated.size),
            "unlabeled": int(split.unlabeled.size),
        },
        "labeled_class_counts": class_count_map(split.labeled, targets),
        "validation_class_counts": class_count_map(split.validation, targets),
    }
    atomic_json_dump(split_summary, destination / "split_summary.json")

    artifacts = train(
        config,
        data,
        destination,
        device,
        resolved_config_hash=resolved_hash,
        resume=resume,
    )
    validation_result = evaluate(artifacts.ema.model, data.validation_loader, device)
    test_result = evaluate(artifacts.ema.model, data.test_loader, device)
    np.savez_compressed(
        destination / "test_predictions.npz",
        probabilities=test_result.probabilities,
        targets=test_result.targets,
        indices=test_result.indices,
    )
    diagnostics = _latest_history(destination / "history.csv")
    metrics = {
        "run_name": destination.name,
        "method": config["method"]["name"],
        "label_ratio": float(config["data"]["label_ratio"]),
        "seed": seed,
        "tau": float(config["method"]["tau"]),
        "lambda_u": float(config["method"]["lambda_u"]),
        "active_strategy": config["active"].get("strategy", "none"),
        "config_hash": config_hash(config),
        "resolved_hash": resolved_hash,
        "split_hash": split.split_hash,
        "split_counts": split_summary["counts"],
        "normalization": {"mean": data.mean, "std": data.std},
        "model_parameters": parameter_count(artifacts.model),
        "device": str(device),
        "execution": {"prefetch_workers": artifacts.prefetch_workers, "torch_threads": torch.get_num_threads()},
        "package_versions": package_versions(),
        "train_steps": artifacts.final_step,
        "runtime_seconds_this_process": artifacts.runtime_seconds,
        "resumed": artifacts.resumed,
        "diagnostics": diagnostics,
        "validation": validation_result.metrics,
        "test": test_result.metrics,
    }
    atomic_json_dump(metrics, metrics_path)
    return RunResult(metrics=metrics, model=artifacts.ema.model, data=data, run_dir=destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Fashion-MNIST experiment")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--method", choices=["supervised", "pseudolabel", "fixmatch"])
    parser.add_argument("--label-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--active-strategy")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(
        load_config(args.config),
        method=args.method,
        label_ratio=args.label_ratio,
        seed=args.seed,
        tau=args.tau,
        device=args.device,
        output_dir=args.output_dir,
        active_strategy=args.active_strategy,
    )
    result = run_experiment(
        config,
        resume=not args.no_resume,
        force=args.force,
    )
    summary = {
        "run_dir": str(result.run_dir.resolve()),
        "test_accuracy": result.metrics["test"]["accuracy"],
        "test_macro_f1": result.metrics["test"]["macro_f1"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
