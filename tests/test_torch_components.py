from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.config import deep_merge, load_config
from src.data import UnlabeledDataset, build_base_datasets, build_data_bundle, build_transforms
from src.engine import train
from src.methods.pseudolabel import pseudolabel_step
from src.metrics import metrics_from_confusion
from src.models import FashionCNN
from src.utils import seed_everything


def test_model_shape_feature_interface_and_parameter_budget() -> None:
    model = FashionCNN()
    logits, features = model(torch.randn(4, 1, 28, 28), return_features=True)
    assert logits.shape == (4, 10)
    assert features.shape == (4, 256)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000 < parameter_count < 1_500_000


def test_unlabeled_dataset_does_not_return_hidden_target() -> None:
    config = load_config("configs/smoke.yaml")
    train_base, _ = build_base_datasets(config)
    mean = float(train_base.data.float().mean() / 255.0)
    std = float(train_base.data.float().std(unbiased=False) / 255.0)
    weak, strong, _ = build_transforms(mean, std)
    dataset = UnlabeledDataset(train_base, np.array([17]), weak, strong)
    item = dataset[0]
    assert len(item) == 3
    weak_image, strong_image, index = item
    assert weak_image.shape == strong_image.shape == (1, 28, 28)
    assert index == 17


def test_pseudo_label_threshold_mask() -> None:
    class ConstantModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(10))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.bias.unsqueeze(0).expand(images.shape[0], -1)

    model = ConstantModel()
    labeled = (torch.zeros(2, 1, 28, 28), torch.tensor([0, 1]), torch.tensor([0, 1]))
    unlabeled = (
        torch.zeros(4, 1, 28, 28),
        torch.zeros(4, 1, 28, 28),
        torch.tensor([2, 3, 4, 5]),
    )
    diagnostic = torch.zeros(10, dtype=torch.long)
    _, strict = pseudolabel_step(
        model,
        labeled,
        unlabeled,
        device=torch.device("cpu"),
        tau=0.95,
        lambda_u=1.0,
        diagnostic_targets=diagnostic,
    )
    _, open_mask = pseudolabel_step(
        model,
        labeled,
        unlabeled,
        device=torch.device("cpu"),
        tau=0.0,
        lambda_u=1.0,
        diagnostic_targets=diagnostic,
    )
    assert strict["accepted"] == 0
    assert open_mask["accepted"] == 4


def test_confusion_metrics() -> None:
    matrix = np.array([[8, 2], [1, 9]])
    metrics = metrics_from_confusion(matrix)
    assert metrics["accuracy"] == pytest.approx(0.85)
    assert 0.0 < metrics["macro_f1"] < 1.0


@pytest.mark.parametrize("device_name,method", [("cpu", "supervised"), ("cuda", "fixmatch")])
def test_deterministic_checkpoint_resume_matches_continuous(device_name: str, method: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    base = load_config("configs/smoke.yaml")
    config = deep_merge(
        base,
        {
            "method": {"name": method},
            "train": {"steps": 4, "eval_interval": 2, "seed": 11, "deterministic": True, "amp": device_name == "cuda"},
        },
    )
    config_hash = "resume-test"
    test_root = Path("outputs") / "test-resume" / uuid4().hex

    seed_everything(11)
    continuous_data = build_data_bundle(config)
    continuous = train(
        config,
        continuous_data,
        test_root / "continuous",
        torch.device(device_name),
        resolved_config_hash=config_hash,
        resume=False,
    )

    seed_everything(11)
    resumed_data = build_data_bundle(config)
    train(
        config,
        resumed_data,
        test_root / "resumed",
        torch.device(device_name),
        resolved_config_hash=config_hash,
        resume=False,
        stop_after=2,
    )
    resumed = train(
        config,
        resumed_data,
        test_root / "resumed",
        torch.device(device_name),
        resolved_config_hash=config_hash,
        resume=True,
    )

    assert resumed.resumed
    for name, expected in continuous.ema.model.state_dict().items():
        actual = resumed.ema.model.state_dict()[name]
        assert torch.equal(actual, expected), name


def test_completed_run_rejects_changed_configuration(tmp_path) -> None:
    from src.run import run_experiment

    config = deep_merge(load_config("configs/smoke.yaml"), {"train": {"device": "cpu", "steps": 1}})
    run_experiment(config, run_dir=tmp_path / "run")
    changed = deep_merge(config, {"train": {"learning_rate": 0.001}})
    with pytest.raises(RuntimeError, match="Completed run configuration"):
        run_experiment(changed, run_dir=tmp_path / "run")
