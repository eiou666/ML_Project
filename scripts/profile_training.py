"""Measure data preparation and CUDA compute separately on an isolated real-data run."""
from pathlib import Path
import os
import sys
import time
import json
import statistics
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from src.config import load_config, deep_merge
from src.data import build_data_bundle
from src.engine import DeterministicBatchStream, _step_seed, build_model, build_optimizer_and_scheduler, ExponentialMovingAverage
from src.methods import fixmatch_step
from src.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=35)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = deep_merge(load_config("configs/local_cuda.yaml"), {"method": {"name": "fixmatch"}, "data": {"label_ratio": 0.01, "num_workers": 0}})
    seed_everything(0)
    data = build_data_bundle(config)
    labeled = DeterministicBatchStream(data.labeled_loader.dataset, 64, 101)
    unlabeled = DeterministicBatchStream(data.unlabeled_loader.dataset, 192, 211)
    model = build_model(config).cuda().train()
    ema = ExponentialMovingAverage(model, 0.999)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler("cuda")
    prepared_iterator = None
    if args.workers:
        from src.prefetch import prepared_steps
        prepared_iterator = iter(prepared_steps(labeled, unlabeled, seed=0, start=0, end=args.steps, workers=args.workers, pin_memory=True))
    rows = []
    started = time.perf_counter()
    for step in range(args.steps):
        _step_seed(0, step)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if prepared_iterator is None:
            lb = labeled.batch(step)
            ub = unlabeled.batch(step)
        else:
            prepared = next(prepared_iterator)
            prepared.restore_cpu_rng()
            lb, ub = prepared.labeled, prepared.unlabeled
        t1 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            loss, _ = fixmatch_step(model, lb, ub, device=torch.device("cuda"), tau=.95, lambda_u=1., diagnostic_targets=data.diagnostic_targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if step >= 5:
            rows.append({"data_ms": (t1-t0)*1000, "compute_ms": (t2-t1)*1000, "total_ms": (t2-t0)*1000})
    result = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
    result.update(workers=args.workers, steps=args.steps, wall_seconds=time.perf_counter()-started)
    result["note"] = "Compute includes transfer and host overhead; timings exclude the first five warmup steps. Check concurrent system load separately."
    output = ROOT / "outputs/performance"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"profile-workers{args.workers}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
