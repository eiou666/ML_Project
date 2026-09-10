from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.integrate import trapezoid
import yaml


METHOD_LABELS = {
    "supervised": "Supervised",
    "pseudolabel": "Pseudo-label",
    "fixmatch": "FixMatch",
}
ACTIVE_LABELS = {
    "random": "Random",
    "entropy": "Entropy",
    "uncertainty_diversity": "Uncertainty + diversity",
}


def _metric_files(input_path: Path) -> list[Path]:
    if input_path.name == "outputs":
        roots = [input_path / "runs", input_path / "active"]
    elif input_path.name == "runs":
        roots = [input_path, input_path.parent / "active"]
    else:
        roots = [input_path]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("metrics.json"))
    return sorted(set(files))


def load_runs(input_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _metric_files(input_path):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        diagnostics = payload.get("diagnostics", {})
        rows.append(
            {
                "path": str(path.parent),
                "run_family": "active" if "active" in {part.lower() for part in path.parts} else "main",
                "run_name": payload["run_name"],
                "method": payload["method"],
                "label_ratio": float(payload["label_ratio"]),
                "seed": int(payload["seed"]),
                "tau": float(payload["tau"]),
                "active_strategy": payload.get("active_strategy", "none"),
                "accuracy": float(payload["test"]["accuracy"]),
                "macro_f1": float(payload["test"]["macro_f1"]),
                "validation_accuracy": float(payload["validation"]["accuracy"]),
                "pseudo_coverage": float(diagnostics.get("pseudo_coverage", 0.0) or 0.0),
                "pseudo_accuracy": float(diagnostics.get("pseudo_accuracy", 0.0) or 0.0),
                "runtime_seconds": float(payload.get("runtime_seconds_this_process", 0.0)),
                "confusion_matrix": payload["test"]["confusion_matrix"],
            }
        )
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found below {input_path}")
    return pd.DataFrame(rows)


def confidence_interval(values: pd.Series, confidence: float = 0.95) -> float:
    clean = values.dropna().astype(float)
    if len(clean) < 2:
        return math.nan
    standard_error = stats.sem(clean)
    return float(standard_error * stats.t.ppf((1 + confidence) / 2, len(clean) - 1))


def summarize_main(runs: pd.DataFrame) -> pd.DataFrame:
    main = runs[(runs["active_strategy"] == "none") & (runs["run_family"] == "main")].copy()
    summary = (
        main.groupby(["method", "label_ratio", "tau"], as_index=False)
        .agg(
            runs=("accuracy", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            pseudo_coverage_mean=("pseudo_coverage", "mean"),
            pseudo_accuracy_mean=("pseudo_accuracy", "mean"),
        )
        .sort_values(["label_ratio", "method", "tau"], ascending=[False, True, True])
    )
    ci_values = []
    for _, row in summary.iterrows():
        condition = main[
            (main["method"] == row["method"])
            & np.isclose(main["label_ratio"], row["label_ratio"])
            & np.isclose(main["tau"], row["tau"])
        ]
        ci_values.append(confidence_interval(condition["accuracy"]))
    summary["accuracy_ci95_half_width"] = ci_values
    full = summary[
        (summary["method"] == "supervised") & np.isclose(summary["label_ratio"], 1.0)
    ]
    full_accuracy = float(full.iloc[0]["accuracy_mean"]) if not full.empty else math.nan
    summary["performance_retention"] = summary["accuracy_mean"] / full_accuracy
    return summary


def paired_ssl_gain(runs: pd.DataFrame) -> pd.DataFrame:
    main = runs[
        (runs["active_strategy"] == "none")
        & (runs["run_family"] == "main")
        & np.isclose(runs["tau"], 0.95)
    ]
    supervised = main[main["method"] == "supervised"][
        ["label_ratio", "seed", "accuracy"]
    ].rename(columns={"accuracy": "supervised_accuracy"})
    fixmatch = main[main["method"] == "fixmatch"][["label_ratio", "seed", "accuracy"]].rename(
        columns={"accuracy": "fixmatch_accuracy"}
    )
    pairs = fixmatch.merge(supervised, on=["label_ratio", "seed"], how="inner")
    if pairs.empty:
        return pd.DataFrame(
            columns=["label_ratio", "pairs", "gain_mean", "gain_std", "gain_ci95_half_width"]
        )
    pairs["gain"] = pairs["fixmatch_accuracy"] - pairs["supervised_accuracy"]
    rows = []
    for ratio, group in pairs.groupby("label_ratio"):
        rows.append(
            {
                "label_ratio": ratio,
                "pairs": len(group),
                "gain_mean": group["gain"].mean(),
                "gain_std": group["gain"].std(),
                "gain_ci95_half_width": confidence_interval(group["gain"]),
            }
        )
    return pd.DataFrame(rows).sort_values("label_ratio")


def load_active_history(input_path: Path) -> pd.DataFrame:
    if input_path.name == "outputs":
        search_root = input_path / "active"
    elif input_path.name == "runs":
        search_root = input_path.parent / "active"
    else:
        search_root = input_path
    files = sorted(search_root.rglob("active_history.csv")) if search_root.exists() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True).drop_duplicates(
        ["seed", "strategy", "round"], keep="last"
    )


def summarize_active(active: pd.DataFrame) -> pd.DataFrame:
    if active.empty:
        return active
    summary = (
        active.groupby(["strategy", "label_ratio"], as_index=False)
        .agg(
            runs=("test_accuracy", "count"),
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            macro_f1_mean=("test_macro_f1", "mean"),
        )
        .sort_values(["strategy", "label_ratio"])
    )
    areas = []
    for strategy, group in active.groupby("strategy"):
        per_seed = []
        for _, seed_group in group.groupby("seed"):
            ordered = seed_group.sort_values("label_ratio")
            per_seed.append(trapezoid(ordered["test_accuracy"], ordered["label_ratio"]))
        areas.append({"strategy": strategy, "aulc_mean": np.mean(per_seed), "aulc_std": np.std(per_seed, ddof=1) if len(per_seed) > 1 else 0.0})
    return summary.merge(pd.DataFrame(areas), on="strategy", how="left")


def _plot_learning_curve(runs: pd.DataFrame, output: Path) -> None:
    data = runs[
        (runs["active_strategy"] == "none")
        & (runs["run_family"] == "main")
        & np.isclose(runs["tau"], 0.95)
    ].copy()
    data["label_percent"] = data["label_ratio"] * 100
    data["Method"] = data["method"].map(METHOD_LABELS)
    plt.figure(figsize=(7.2, 4.8))
    sns.lineplot(data=data, x="label_percent", y="accuracy", hue="Method", marker="o", errorbar="sd")
    plt.xscale("log")
    plt.xticks([1, 5, 10, 100], ["1", "5", "10", "100"])
    plt.xlabel("Label budget (%)")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "label_efficiency_curve.png", dpi=220)
    plt.close()


def _plot_gain(gain: pd.DataFrame, output: Path) -> None:
    if gain.empty:
        return
    plt.figure(figsize=(6.4, 4.4))
    x = np.arange(len(gain))
    plt.bar(x, gain["gain_mean"], yerr=gain["gain_ci95_half_width"], capsize=5)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xticks(x, [f"{ratio * 100:g}%" for ratio in gain["label_ratio"]])
    plt.ylabel("Paired accuracy gain: FixMatch - supervised")
    plt.xlabel("Label budget")
    plt.tight_layout()
    plt.savefig(output / "paired_ssl_gain.png", dpi=220)
    plt.close()


def _plot_confusions(runs: pd.DataFrame, output: Path) -> None:
    candidates = runs[
        (runs["seed"] == 0)
        & np.isclose(runs["label_ratio"], 0.01)
        & np.isclose(runs["tau"], 0.95)
        & runs["method"].isin(["supervised", "fixmatch"])
        & (runs["active_strategy"] == "none")
        & (runs["run_family"] == "main")
    ]
    if candidates["method"].nunique() < 2:
        return
    labels = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, method in zip(axes, ["supervised", "fixmatch"]):
        row = candidates[candidates["method"] == method].iloc[0]
        matrix = np.asarray(row["confusion_matrix"], dtype=float)
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        sns.heatmap(matrix, ax=axis, cmap="Blues", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
        axis.set_title(f"{METHOD_LABELS[method]} - 1% labels")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    figure.tight_layout()
    figure.savefig(output / "confusion_1pct.png", dpi=220)
    plt.close(figure)


def _plot_ablation(runs: pd.DataFrame, output: Path) -> None:
    data = runs[
        (runs["method"] == "fixmatch")
        & np.isclose(runs["label_ratio"], 0.01)
        & (runs["active_strategy"] == "none")
        & (runs["run_family"] == "main")
    ]
    if data["tau"].nunique() < 2:
        return
    common_seeds = set.intersection(*(set(group["seed"]) for _, group in data.groupby("tau")))
    data = data[data["seed"].isin(common_seeds)]
    if data.empty:
        return
    summary = data.groupby("tau", as_index=False).agg(
        accuracy=("accuracy", "mean"),
        coverage=("pseudo_coverage", "mean"),
        pseudo_accuracy=("pseudo_accuracy", "mean"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for axis, column, title in zip(
        axes,
        ["coverage", "pseudo_accuracy", "accuracy"],
        ["Pseudo-label coverage", "Pseudo-label accuracy", "Test accuracy"],
    ):
        axis.plot(summary["tau"], summary[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Confidence threshold")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "threshold_ablation.png", dpi=220)
    plt.close(figure)


def _plot_active(active_summary: pd.DataFrame, output: Path) -> None:
    if active_summary.empty:
        return
    data = active_summary.copy()
    data["Label budget (%)"] = data["label_ratio"] * 100
    data["Strategy"] = data["strategy"].map(ACTIVE_LABELS)
    plt.figure(figsize=(7.0, 4.6))
    for strategy, group in data.groupby("Strategy"):
        ordered = group.sort_values("Label budget (%)")
        plt.errorbar(
            ordered["Label budget (%)"],
            ordered["accuracy_mean"],
            yerr=ordered["accuracy_std"].fillna(0),
            marker="o",
            capsize=4,
            label=strategy,
        )
    plt.xticks([1, 2, 3])
    plt.xlabel("Label budget (%)")
    plt.ylabel("Test accuracy")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "active_learning_curve.png", dpi=220)
    plt.close()


def _plot_representative_examples(runs: pd.DataFrame, output: Path) -> None:
    candidates = runs[
        (runs["run_family"] == "main")
        & (runs["method"] == "fixmatch")
        & (runs["seed"] == 0)
        & np.isclose(runs["label_ratio"], 0.01)
        & np.isclose(runs["tau"], 0.95)
    ]
    if candidates.empty:
        return
    run_dir = Path(candidates.iloc[0]["path"])
    prediction_path = run_dir / "test_predictions.npz"
    config_path = run_dir / "resolved_config.yaml"
    if not prediction_path.exists() or not config_path.exists():
        return
    try:
        from torchvision.datasets import FashionMNIST

        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        dataset = FashionMNIST(root=config["data"]["root"], train=False, download=False)
    except Exception as exc:
        print(f"Skipping representative examples: {exc}")
        return

    arrays = np.load(prediction_path)
    probabilities = arrays["probabilities"]
    targets = arrays["targets"].astype(np.int64)
    indices = arrays["indices"].astype(np.int64)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    correct_positions = np.flatnonzero(predictions == targets)
    error_positions = np.flatnonzero(predictions != targets)
    success = correct_positions[np.argsort(-confidence[correct_positions])[:4]]
    failure = error_positions[np.argsort(-confidence[error_positions])[:4]] if error_positions.size else correct_positions[:4]
    uncertain = np.argsort(-entropy)[:4]
    groups = [("High-confidence correct", success), ("High-confidence error", failure), ("Highest uncertainty", uncertain)]
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    figure, axes = plt.subplots(3, 4, figsize=(10.5, 7.2))
    for row, (group_name, positions) in enumerate(groups):
        for column, position in enumerate(positions):
            axis = axes[row, column]
            image_index = int(indices[position])
            axis.imshow(dataset.data[image_index].numpy(), cmap="gray")
            axis.set_title(
                f"T:{class_names[targets[position]]}\nP:{class_names[predictions[position]]} ({confidence[position]:.2f})",
                fontsize=8,
            )
            axis.axis("off")
        axes[row, 0].set_ylabel(group_name, fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "representative_examples.png", dpi=220)
    plt.close(figure)


def _write_markdown_table(summary: pd.DataFrame, output: Path) -> None:
    table = summary.copy()
    table["labels"] = table["label_ratio"].map(lambda value: f"{value * 100:g}%")
    table["method_name"] = table["method"].map(METHOD_LABELS)
    table["accuracy"] = table.apply(
        lambda row: f"{row['accuracy_mean']:.4f} ± {0.0 if pd.isna(row['accuracy_std']) else row['accuracy_std']:.4f}",
        axis=1,
    )
    columns = ["method_name", "labels", "tau", "runs", "accuracy", "macro_f1_mean", "performance_retention"]
    with (output / "main_results.md").open("w", encoding="utf-8") as handle:
        handle.write("# Main experimental results\n\n")
        handle.write(table[columns].to_markdown(index=False))
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics and generate figures")
    parser.add_argument("--input", default="outputs")
    parser.add_argument("--output", default="outputs/summary")
    args = parser.parse_args()
    input_path = Path(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    runs = load_runs(input_path)
    summary = summarize_main(runs)
    gain = paired_ssl_gain(runs)
    active = load_active_history(input_path)
    active_summary = summarize_active(active)

    runs.drop(columns=["confusion_matrix"]).to_csv(output / "runs.csv", index=False)
    summary.to_csv(output / "main_summary.csv", index=False)
    gain.to_csv(output / "paired_ssl_gain.csv", index=False)
    if not active.empty:
        active.to_csv(output / "active_runs.csv", index=False)
        active_summary.to_csv(output / "active_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_learning_curve(runs, output)
    _plot_gain(gain, output)
    _plot_confusions(runs, output)
    _plot_ablation(runs, output)
    _plot_active(active_summary, output)
    _plot_representative_examples(runs, output)
    _write_markdown_table(summary, output)
    print(json.dumps({"runs": len(runs), "output": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
