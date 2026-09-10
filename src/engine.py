from __future__ import annotations

import copy
import gc
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, default_collate
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .data import DataBundle
from .methods import fixmatch_step, pseudolabel_step, supervised_step
from .metrics import evaluate
from .models import FashionCNN
from .utils import append_csv_row, capture_rng_state, restore_rng_state


HISTORY_FIELDS = [
    "step",
    "learning_rate",
    "supervised_loss",
    "unsupervised_loss",
    "train_accuracy",
    "pseudo_coverage",
    "pseudo_accuracy",
    "pseudo_class_counts",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "elapsed_seconds",
]


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        source_parameters = dict(source.named_parameters())
        for name, parameter in self.model.named_parameters():
            parameter.mul_(self.decay).add_(source_parameters[name], alpha=1.0 - self.decay)
        source_buffers = dict(source.named_buffers())
        for name, buffer in self.model.named_buffers():
            buffer.copy_(source_buffers[name])

    def state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict(), "decay": self.decay}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.decay = float(state["decay"])


def _stack_local_tensors(batch, *, collate_fn_map=None):
    return torch.stack(batch, 0)


class DeterministicBatchStream:
    """Step-addressable batches make checkpoint resume independent of iterator state."""

    def __init__(self, dataset: Dataset, batch_size: int, seed: int) -> None:
        if len(dataset) < 1:
            raise ValueError("Training dataset cannot be empty")
        self.dataset = dataset
        self.batch_size = min(int(batch_size), len(dataset))
        self.seed = int(seed)
        self.steps_per_epoch = max(1, len(dataset) // self.batch_size)

    def batch(self, step: int, *, shared_memory: bool = True) -> Any:
        epoch, batch_index = divmod(step, self.steps_per_epoch)
        rng = np.random.default_rng(self.seed + epoch * 104729)
        order = rng.permutation(len(self.dataset))
        start = batch_index * self.batch_size
        positions = order[start : start + self.batch_size]
        examples = [self.dataset[int(position)] for position in positions]
        if shared_memory:
            return default_collate(examples)
        # Windows workers must not allocate a new named tensor mapping per batch.
        local_map = dict(default_collate_fn_map)
        local_map[torch.Tensor] = _stack_local_tensors
        return collate(examples, collate_fn_map=local_map)


class CyclingLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> Any:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


@dataclass
class TrainingArtifacts:
    model: nn.Module
    ema: ExponentialMovingAverage
    final_step: int
    runtime_seconds: float
    resumed: bool
    prefetch_workers: int = 0


def build_model(config: dict[str, Any]) -> FashionCNN:
    return FashionCNN(
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
    )


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, float(step + 1) / float(warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer_and_scheduler(
    model: nn.Module, config: dict[str, Any]
) -> tuple[AdamW, LambdaLR]:
    train_config = config["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    total_steps = int(train_config["steps"])
    warmup_steps = int(train_config["warmup_steps"])
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_multiplier(step, total_steps, warmup_steps),
    )
    return optimizer, scheduler


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    config_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": int(step),
            "config_hash": config_hash,
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    config_hash: str,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config_hash") != config_hash:
        raise RuntimeError("Checkpoint configuration hash does not match the resolved run config")
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["step"])


def _step_seed(seed: int, step: int) -> None:
    value = (seed * 1_000_003 + step * 97_409 + 17) % (2**31 - 1)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def train(
    config: dict[str, Any],
    data: DataBundle,
    run_dir: Path,
    device: torch.device,
    *,
    resolved_config_hash: str,
    resume: bool = True,
    stop_after: int | None = None,
) -> TrainingArtifacts:
    train_config = config["train"]
    method = str(config["method"]["name"])
    total_steps = int(train_config["steps"])
    end_step = min(total_steps, int(stop_after)) if stop_after is not None else total_steps
    if end_step <= 0:
        raise ValueError("stop_after must be positive when provided")
    deterministic = bool(train_config.get("deterministic", True))

    model = build_model(config).to(device)
    ema = ExponentialMovingAverage(model, float(train_config["ema_decay"]))
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    start_step = 0
    resumed = False
    if resume and checkpoint_path.exists():
        start_step = _load_checkpoint(
            checkpoint_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_hash=resolved_config_hash,
            device=device,
        )
        resumed = start_step > 0

    if deterministic:
        labeled_stream: Any = DeterministicBatchStream(
            data.labeled_loader.dataset,
            int(train_config["batch_labeled"]),
            int(train_config["seed"]) + 101,
        )
        unlabeled_stream: Any = None
        if data.unlabeled_loader is not None:
            unlabeled_stream = DeterministicBatchStream(
                data.unlabeled_loader.dataset,
                int(train_config["batch_unlabeled"]),
                int(train_config["seed"]) + 211,
            )
    else:
        labeled_stream = CyclingLoader(data.labeled_loader)
        unlabeled_stream = CyclingLoader(data.unlabeled_loader) if data.unlabeled_loader else None

    prefetch_workers = int(os.environ.get("ML_PREFETCH_WORKERS", "0")) if deterministic else 0
    prepared_iterator = None
    if prefetch_workers > 0 and start_step < end_step:
        from .prefetch import prepared_steps
        prepared_iterator = iter(prepared_steps(
            labeled_stream, unlabeled_stream, seed=int(train_config["seed"]),
            start=start_step, end=end_step, workers=prefetch_workers,
            pin_memory=device.type == "cuda" and bool(config["data"].get("pin_memory", True)),
        ))
    print(f"data preparation workers={prefetch_workers} deterministic={deterministic}", flush=True)

    accumulators = {
        "supervised_loss": 0.0,
        "unsupervised_loss": 0.0,
        "train_accuracy": 0.0,
        "accepted": 0.0,
        "unlabeled_seen": 0.0,
        "pseudo_correct": 0.0,
        "batches": 0.0,
    }
    pseudo_class_counts = np.zeros(int(config["model"]["num_classes"]), dtype=np.int64)
    started = time.perf_counter()
    model.train()

    for zero_based_step in range(start_step, end_step):
        _step_seed(int(train_config["seed"]), zero_based_step)
        if prepared_iterator is not None:
            prepared = next(prepared_iterator)
            prepared.restore_cpu_rng()
            labeled_batch, unlabeled_batch = prepared.labeled, prepared.unlabeled
        elif deterministic:
            labeled_batch = labeled_stream.batch(zero_based_step)
            unlabeled_batch = unlabeled_stream.batch(zero_based_step) if unlabeled_stream else None
        else:
            labeled_batch = labeled_stream.next()
            unlabeled_batch = unlabeled_stream.next() if unlabeled_stream else None

        optimizer.zero_grad(set_to_none=True)
        autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=amp_enabled):
            if method == "supervised":
                loss, batch_stats = supervised_step(model, labeled_batch, device)
            else:
                if unlabeled_batch is None:
                    raise RuntimeError("SSL method was started without an unlabeled loader")
                step_function = fixmatch_step if method == "fixmatch" else pseudolabel_step
                loss, batch_stats = step_function(
                    model,
                    labeled_batch,
                    unlabeled_batch,
                    device=device,
                    tau=float(config["method"]["tau"]),
                    lambda_u=float(config["method"]["lambda_u"]),
                    diagnostic_targets=data.diagnostic_targets,
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_config["gradient_clip_norm"])
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        for key in accumulators:
            if key == "batches":
                accumulators[key] += 1.0
            else:
                accumulators[key] += float(batch_stats[key])
        pseudo_class_counts += np.asarray(batch_stats["pseudo_class_counts"], dtype=np.int64)

        step = zero_based_step + 1
        if step == start_step + 1 or step % int(train_config.get("log_interval", 100)) == 0:
            print(
                f"[{method}] step={step}/{total_steps} "
                f"loss={float(loss.detach()):.4f} "
                f"elapsed={time.perf_counter() - started:.1f}s device={device}",
                flush=True,
            )
            gc.collect()
        should_evaluate = step % int(train_config["eval_interval"]) == 0 or step == end_step
        if should_evaluate:
            validation = evaluate(ema.model, data.validation_loader, device).metrics
            print(f"validation step={step} accuracy={validation['accuracy']:.4f} loss={validation['loss']:.4f}", flush=True)
            batches = max(1.0, accumulators["batches"])
            accepted = accumulators["accepted"]
            row = {
                "step": step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "supervised_loss": accumulators["supervised_loss"] / batches,
                "unsupervised_loss": accumulators["unsupervised_loss"] / batches,
                "train_accuracy": accumulators["train_accuracy"] / batches,
                "pseudo_coverage": accepted / max(1.0, accumulators["unlabeled_seen"]),
                "pseudo_accuracy": accumulators["pseudo_correct"] / max(1.0, accepted),
                "pseudo_class_counts": ";".join(str(value) for value in pseudo_class_counts.tolist()),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_macro_f1": validation["macro_f1"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            append_csv_row(run_dir / "history.csv", row, HISTORY_FIELDS)
            _save_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                config_hash=resolved_config_hash,
            )
            for key in accumulators:
                accumulators[key] = 0.0
            pseudo_class_counts.fill(0)
            gc.collect()
            model.train()

    del prepared_iterator
    gc.collect()
    return TrainingArtifacts(
        model=model,
        ema=ema,
        final_step=end_step,
        runtime_seconds=time.perf_counter() - started,
        resumed=resumed,
        prefetch_workers=prefetch_workers,
    )


def load_ema_model_from_checkpoint(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> nn.Module:
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"]["model"])
    model.eval()
    return model
