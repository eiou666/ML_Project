from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    module: str
    arguments: tuple[str, ...]

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.arguments]


def build_jobs(suite: str, config: str, seeds: list[int], device: str | None) -> list[Job]:
    jobs: list[Job] = []
    device_args = ("--device", device) if device else ()
    if suite == "main":
        conditions = [
            ("supervised", 1.00),
            ("supervised", 0.10),
            ("supervised", 0.05),
            ("supervised", 0.01),
            ("pseudolabel", 0.10),
            ("pseudolabel", 0.05),
            ("pseudolabel", 0.01),
            ("fixmatch", 0.10),
            ("fixmatch", 0.05),
            ("fixmatch", 0.01),
        ]
        for method, ratio in conditions:
            for seed in seeds:
                jobs.append(
                    Job(
                        "src.run",
                        (
                            "--config",
                            config,
                            "--method",
                            method,
                            "--label-ratio",
                            str(ratio),
                            "--seed",
                            str(seed),
                            *device_args,
                        ),
                    )
                )
    elif suite == "ablation":
        for tau in (0.0, 0.80, 0.95):
            for seed in seeds:
                jobs.append(
                    Job(
                        "src.run",
                        (
                            "--config",
                            config,
                            "--method",
                            "fixmatch",
                            "--label-ratio",
                            "0.01",
                            "--tau",
                            str(tau),
                            "--seed",
                            str(seed),
                            *device_args,
                        ),
                    )
                )
    elif suite == "active_learning":
        for seed in seeds:
            jobs.append(
                Job(
                    "src.active",
                    (
                        "--config",
                        config,
                        "--seed",
                        str(seed),
                        *device_args,
                    ),
                )
            )
    elif suite == "smoke":
        for method in ("supervised", "pseudolabel", "fixmatch"):
            jobs.append(
                Job(
                    "src.run",
                    (
                        "--config",
                        config,
                        "--method",
                        method,
                        "--label-ratio",
                        "0.10",
                        "--seed",
                        "0",
                        *device_args,
                    ),
                )
            )
    else:
        raise ValueError(f"Unknown suite: {suite}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable experiment suite")
    parser.add_argument(
        "--suite",
        required=True,
        choices=["main", "ablation", "active_learning", "smoke"],
    )
    parser.add_argument("--config")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()

    default_config = "configs/smoke.yaml" if args.suite == "smoke" else "configs/main.yaml"
    config = args.config or default_config
    default_seeds = [0, 1, 2] if args.suite in {"ablation", "active_learning"} else [0, 1, 2, 3, 4]
    seeds = args.seeds or default_seeds
    jobs = build_jobs(args.suite, config, seeds, args.device)
    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]

    if args.dry_run:
        print(json.dumps([job.command() for job in jobs], ensure_ascii=False, indent=2))
        return

    failures: list[dict[str, object]] = []
    for index, job in enumerate(jobs, start=1):
        command = job.command()
        if args.force:
            command.append("--force")
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append({"command": command, "returncode": completed.returncode})
            if not args.keep_going:
                raise SystemExit(completed.returncode)
    if failures:
        print(json.dumps({"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

