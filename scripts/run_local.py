"""Sequential local GPU queue with durable logs, status and a Windows process lock."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs" / "local-training"


def write_status(**values) -> None:
    values.update(pid=os.getpid(), updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    temporary = LOGS / "status.json.tmp"
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, LOGS / "status.json")


def main() -> None:
    os.chdir(ROOT)
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS / "queue.lock").open("a+b")
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("A local training queue is already running")
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", ML_TORCH_THREADS="4")
    env.setdefault("ML_PREFETCH_WORKERS", "4")
    env["TEMP"] = env["TMP"] = str(ROOT / "tmp" / "training")
    Path(env["TEMP"]).mkdir(parents=True, exist_ok=True)
    phases = [
        (suite, ["-m", "src.sweep", "--suite", suite, "--config", "configs/local_cuda.yaml", "--device", "cuda"])
        for suite in ("main", "ablation", "active_learning")
    ]
    phases.append(("aggregate", ["-m", "src.aggregate", "--input", "outputs", "--output", "outputs/summary"]))
    completed = []
    phase = "starting"
    try:
        for phase, arguments in phases:
            log_path = LOGS / f"{phase}.log"
            command = [sys.executable, "-u", *arguments]
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nStarted {time.strftime('%Y-%m-%d %H:%M:%S')}: {command!r}\n")
                handle.flush()
                process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
                write_status(state="running", phase=phase, child_pid=process.pid, completed=completed, log=str(log_path), command=command)
                code = process.wait()
            if code:
                write_status(state="failed", phase=phase, returncode=code, completed=completed, log=str(log_path))
                raise SystemExit(code)
            completed.append(phase)
        write_status(state="complete", phase="done", completed=completed)
    except Exception as exc:
        write_status(state="failed", phase=phase, error=str(exc), completed=completed)
        raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == "__main__":
    main()
