from pathlib import Path

import pytest

from src.config import apply_overrides, config_hash, load_config


def test_main_config_loads_and_hash_is_stable() -> None:
    config = load_config(Path("configs/main.yaml"))
    assert config["method"]["name"] == "supervised"
    assert config_hash(config) == config_hash(config)


def test_overrides_do_not_mutate_source() -> None:
    config = load_config("configs/main.yaml")
    changed = apply_overrides(config, method="fixmatch", label_ratio=0.01, seed=4)
    assert config["method"]["name"] == "supervised"
    assert changed["method"]["name"] == "fixmatch"
    assert changed["data"]["label_ratio"] == 0.01
    assert changed["train"]["seed"] == 4


def test_invalid_ratio_is_rejected() -> None:
    config = load_config("configs/main.yaml")
    with pytest.raises(ValueError, match="label_ratio"):
        apply_overrides(config, label_ratio=0.0)

