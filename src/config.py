from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ALLOWED_METHODS = {"supervised", "pseudolabel", "fixmatch"}
ALLOWED_ACTIVE_STRATEGIES = {
    "none",
    "random",
    "entropy",
    "uncertainty_diversity",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    validate_config(config)
    return config


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_overrides(
    config: dict[str, Any],
    *,
    method: str | None = None,
    label_ratio: float | None = None,
    seed: int | None = None,
    tau: float | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    active_strategy: str | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if method is not None:
        updates.setdefault("method", {})["name"] = method
    if label_ratio is not None:
        updates.setdefault("data", {})["label_ratio"] = label_ratio
    if seed is not None:
        updates.setdefault("train", {})["seed"] = seed
    if tau is not None:
        updates.setdefault("method", {})["tau"] = tau
    if device is not None:
        updates.setdefault("train", {})["device"] = device
    if output_dir is not None:
        updates.setdefault("project", {})["output_dir"] = output_dir
    if active_strategy is not None:
        updates.setdefault("active", {})["strategy"] = active_strategy
    merged = deep_merge(config, updates)
    validate_config(merged)
    return merged


def validate_config(config: dict[str, Any]) -> None:
    required = {"project", "data", "model", "method", "train", "active"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    method = str(config["method"].get("name", ""))
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unsupported method {method!r}; expected {sorted(ALLOWED_METHODS)}")

    strategy = str(config["active"].get("strategy", "none"))
    if strategy not in ALLOWED_ACTIVE_STRATEGIES:
        raise ValueError(
            f"Unsupported active strategy {strategy!r}; "
            f"expected {sorted(ALLOWED_ACTIVE_STRATEGIES)}"
        )

    ratio = float(config["data"].get("label_ratio", 0.0))
    if not 0.0 < ratio <= 1.0:
        raise ValueError("data.label_ratio must be in (0, 1]")

    validation_fraction = float(config["data"].get("validation_fraction", 0.0))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("data.validation_fraction must be in (0, 1)")

    tau = float(config["method"].get("tau", 0.95))
    if not 0.0 <= tau <= 1.0:
        raise ValueError("method.tau must be in [0, 1]")

    if int(config["train"].get("steps", 0)) <= 0:
        raise ValueError("train.steps must be positive")
    if int(config["train"].get("batch_labeled", 0)) <= 0:
        raise ValueError("train.batch_labeled must be positive")
    if method != "supervised" and int(config["train"].get("batch_unlabeled", 0)) <= 0:
        raise ValueError("train.batch_unlabeled must be positive for SSL")


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ratio_slug(ratio: float) -> str:
    percentage = ratio * 100.0
    if percentage.is_integer():
        return f"{int(percentage):03d}pct"
    return f"{percentage:g}pct".replace(".", "p")


def make_run_name(config: dict[str, Any], *, split_hash: str | None = None) -> str:
    method = config["method"]["name"]
    ratio = ratio_slug(float(config["data"]["label_ratio"]))
    seed = int(config["train"]["seed"])
    tau = float(config["method"]["tau"])
    active = config["active"].get("strategy", "none")
    suffix = split_hash[:8] if split_hash else config_hash(config)[:8]
    pieces = [method, ratio, f"seed{seed}"]
    if method != "supervised":
        pieces.append(f"tau{tau:g}".replace(".", "p"))
    if active != "none":
        pieces.append(str(active))
    pieces.append(suffix)
    return "-".join(pieces)


def dump_yaml(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
