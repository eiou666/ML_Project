import random

import numpy as np
import pytest
import torch

from src.config import deep_merge, load_config
from src.data import build_data_bundle
from src.engine import DeterministicBatchStream, _step_seed, train
from src.prefetch import StepDataset, prepared_steps
from src.utils import seed_everything


def test_worker_payload_avoids_tensor_shared_memory(monkeypatch):
    config = deep_merge(load_config("configs/smoke.yaml"), {"method": {"name": "fixmatch"}})
    data = build_data_bundle(config)
    labeled = DeterministicBatchStream(data.labeled_loader.dataset, 16, 101)
    unlabeled = DeterministicBatchStream(data.unlabeled_loader.dataset, 32, 211)

    def forbidden_mapping(*args, **kwargs):
        raise AssertionError("Per-step shared tensor storage must not be allocated")

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    monkeypatch.setattr(torch.storage.TypedStorage, "_new_shared", forbidden_mapping)
    payload = StepDataset(labeled, unlabeled, 0, 0, 1)[0]
    for batch in (payload.labeled, payload.unlabeled):
        assert all(isinstance(value, np.ndarray) for value in batch)
    assert isinstance(payload.torch_rng, np.ndarray)


def test_prefetched_batches_and_rng_match_serial_across_epoch_boundary():
    config = deep_merge(load_config("configs/smoke.yaml"), {"method": {"name": "fixmatch"}})
    data = build_data_bundle(config)
    labeled = DeterministicBatchStream(data.labeled_loader.dataset, 16, 101)
    unlabeled = DeterministicBatchStream(data.unlabeled_loader.dataset, 32, 211)
    loader = prepared_steps(labeled, unlabeled, seed=0, start=4, end=8, workers=2, pin_memory=False)
    for step, prepared in zip(range(4, 8), loader):
        _step_seed(0, step)
        expected = (labeled.batch(step), unlabeled.batch(step))
        for actual_batch, expected_batch in zip((prepared.labeled, prepared.unlabeled), expected):
            for actual, reference in zip(actual_batch, expected_batch):
                assert torch.equal(actual, reference)
        assert random.getstate() == prepared.python_rng
        np.testing.assert_array_equal(np.random.get_state()[1], prepared.numpy_rng[1])
        assert torch.equal(torch.get_rng_state(), prepared.torch_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_switch_from_serial_checkpoint_to_prefetch_preserves_cuda_training(tmp_path, monkeypatch):
    config = deep_merge(load_config("configs/smoke.yaml"), {
        "method": {"name": "fixmatch", "tau": 0.0},
        "data": {"pin_memory": True},
        "train": {"steps": 6, "eval_interval": 3, "seed": 11, "amp": True, "batch_labeled": 64, "batch_unlabeled": 192},
    })
    monkeypatch.setenv("ML_PREFETCH_WORKERS", "0")
    seed_everything(11)
    continuous = train(config, build_data_bundle(config), tmp_path / "continuous", torch.device("cuda"), resolved_config_hash="prefetch-test", resume=False)
    seed_everything(11)
    data = build_data_bundle(config)
    train(config, data, tmp_path / "resumed", torch.device("cuda"), resolved_config_hash="prefetch-test", resume=False, stop_after=3)
    monkeypatch.setenv("ML_PREFETCH_WORKERS", "4")
    resumed = train(config, data, tmp_path / "resumed", torch.device("cuda"), resolved_config_hash="prefetch-test", resume=True)
    assert resumed.resumed
    for expected_model, actual_model in [(continuous.model, resumed.model), (continuous.ema.model, resumed.ema.model)]:
        for name, expected in expected_model.state_dict().items():
            assert torch.equal(expected, actual_model.state_dict()[name]), name
