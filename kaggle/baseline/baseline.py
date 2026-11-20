from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


WORK_ROOT = Path("/kaggle/working/althia")
OUTPUT = Path("/kaggle/working/results-smoke.jsonl")
METADATA = Path("/kaggle/working/run-metadata.json")


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    input_base = Path("/kaggle/input")
    manifests = sorted(input_base.rglob("smoke-manifest.json"))
    requirement_files = sorted(input_base.rglob("source/requirements.txt"))
    if len(manifests) != 1 or len(requirement_files) != 1:
        raise FileNotFoundError(
            f"attached dataset layout is unexpected: "
            f"manifests={manifests}, requirements={requirement_files}"
        )
    input_root = manifests[0].parent
    source_root = requirement_files[0].parent
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    shutil.copytree(source_root, WORK_ROOT)

    install = run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=WORK_ROOT,
    )
    print(install.stdout, flush=True)
    prepared = run([sys.executable, "prepare.py"], cwd=WORK_ROOT)
    print(prepared.stdout, flush=True)

    started = time.perf_counter()
    completed = run(
        [
            sys.executable,
            "run.py",
            "--input",
            str(input_root),
            "--output",
            str(OUTPUT),
            "--model",
            str(WORK_ROOT / "yolo11l.pt"),
            "--device",
            "0",
            "--batch-size",
            "1",
        ],
        cwd=WORK_ROOT,
    )
    elapsed = time.perf_counter() - started
    print(completed.stdout, flush=True)
    validation = run(
        [
            sys.executable,
            "scripts/validate_jsonl.py",
            str(OUTPUT),
            "--expected-videos",
            "3",
            "--no-errors",
        ],
        cwd=WORK_ROOT,
    )
    print(validation.stdout, flush=True)

    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    ).stdout.strip()
    freeze = run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines()
    manifest = json.loads(
        (input_root / "smoke-manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "elapsed_seconds": elapsed,
        "gpu": gpu,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "source_sha256": source_tree_sha256(source_root),
        "fixture_sha256": manifest["sha256"],
        "fixture_source_rows": manifest["source_rows"],
        "requirements": freeze,
        "validation": json.loads(validation.stdout.strip().splitlines()[-1]),
        "baseline": {
            "model": "yolo11l.pt",
            "imgsz": 640,
            "conf": 0.4,
            "iou": 0.7,
            "batch_size": 1,
            "precision": "fp32",
        },
        "results_sha256": sha256(OUTPUT),
    }
    METADATA.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

