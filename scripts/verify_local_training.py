"""Exercise real-data GPU training at formal batch sizes without producing formal results."""
import json
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from src.config import deep_merge, load_config
from src.run import run_experiment


def main():
    os.chdir(ROOT)
    assert torch.cuda.is_available(), "CUDA unavailable"
    base = load_config("configs/local_cuda.yaml")
    results = []
    for method in ("supervised", "pseudolabel", "fixmatch"):
        config = deep_merge(base, {
            "project": {"output_dir": "outputs/local-validation"},
            "data": {"label_ratio": 0.01},
            "method": {"name": method, "tau": 0.0},
            "train": {"steps": 3, "warmup_steps": 1, "eval_interval": 3, "log_interval": 1},
        })
        torch.cuda.reset_peak_memory_stats()
        result = run_experiment(config)
        assert result.metrics["device"] == "cuda"
        assert result.metrics["train_steps"] == 3
        if method != "supervised":
            assert result.metrics["diagnostics"]["pseudo_coverage"] == 1.0
        results.append({
            "method": method, "steps": result.metrics["train_steps"],
            "device": result.metrics["device"],
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "run_dir": str(result.run_dir),
        })
        del result
        torch.cuda.empty_cache()
    path = ROOT / "outputs" / "environment-check" / "gpu-training-validation.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
