from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


WORK_ROOT = Path("/kaggle/working/althia")
WORKING = Path("/kaggle/working")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, check=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {command}")


def main() -> int:
    input_base = Path("/kaggle/input")
    fixtures = sorted(input_base.rglob("reference.parquet"))
    manifests = sorted(input_base.rglob("reference-manifest.json"))
    requirements = sorted(input_base.rglob("source/requirements.txt"))
    if len(fixtures) != 1 or len(manifests) != 1 or len(requirements) != 1:
        raise FileNotFoundError(
            "unexpected attached dataset layout: "
            f"fixtures={fixtures}, manifests={manifests}, requirements={requirements}"
        )

    fixture = fixtures[0]
    manifest = manifests[0]
    source_root = requirements[0].parent
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    shutil.copytree(source_root, WORK_ROOT)

    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], WORK_ROOT)
    run([sys.executable, "prepare.py"], WORK_ROOT)
    run(
        [
            sys.executable,
            "scripts/probe_reference_fixture.py",
            "--fixture",
            str(fixture),
            "--output",
            str(WORKING / "probe.json"),
        ],
        WORK_ROOT,
    )
    run(
        [
            sys.executable,
            "scripts/generate_reference.py",
            "--fixture-dir",
            str(fixture.parent),
            "--output-jsonl",
            str(WORKING / "expected.jsonl"),
            "--output-metadata",
            str(WORKING / "metadata.json"),
            "--model",
            str(WORK_ROOT / "yolo11l.pt"),
            "--device",
            "0",
            "--manifest",
            str(manifest),
        ],
        WORK_ROOT,
    )
    run(
        [
            sys.executable,
            "scripts/validate_jsonl.py",
            str(WORKING / "expected.jsonl"),
            "--expected-videos",
            "10",
            "--no-errors",
        ],
        WORK_ROOT,
    )
    run(
        [
            sys.executable,
            "scripts/compare_reference.py",
            str(WORKING / "expected.jsonl"),
            "--reference",
            str(WORKING / "expected.jsonl"),
            "--exact",
            "--json",
        ],
        WORK_ROOT,
    )

    probe = json.loads((WORKING / "probe.json").read_text(encoding="utf-8"))
    metadata = json.loads((WORKING / "metadata.json").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "probe_video_count": probe["video_count"],
                "has_nonzero_rotation": probe["has_nonzero_rotation"],
                "summary": metadata["summary"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
