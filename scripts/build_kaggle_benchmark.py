from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_benchmark_script() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "scripts", "tests"]:
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, p.relative_to(ROOT).as_posix())
        for single in ["run.py", "prepare.py", "requirements.txt"]:
            zf.write(ROOT / single, single)

    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

    content = f'''from __future__ import annotations

import base64
import hashlib
import io
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

print("=" * 60, flush=True)
print("ALTHIA T4 BATCH-SIZE BENCHMARK RUNNER", flush=True)
print("=" * 60, flush=True)

EXPECTED_JSONL_SHA256 = "7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d"
REFERENCE_PARQUET_SHA256 = "0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74"
YOLO11L_SHA256 = "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    input_base = Path("/kaggle/input")
    working_base = Path("/kaggle/working")
    work_root = working_base / "althia"

    # 1. Locate authoritative files by SHA-256
    print("Locating input artifacts by SHA-256...", flush=True)
    fixture_path = None
    for cand in sorted(input_base.rglob("*.parquet")):
        if cand.is_file() and file_sha256(cand) == REFERENCE_PARQUET_SHA256:
            fixture_path = cand
            break
    if fixture_path is None:
        raise FileNotFoundError(f"Could not locate reference.parquet matching hash {{REFERENCE_PARQUET_SHA256}}")
    print(f"Verified reference fixture: {{fixture_path}}", flush=True)

    expected_path = None
    for cand in sorted(input_base.rglob("*.jsonl")):
        if cand.is_file() and file_sha256(cand) == EXPECTED_JSONL_SHA256:
            expected_path = cand
            break
    if expected_path is None:
        raise FileNotFoundError(f"Could not locate expected.jsonl matching hash {{EXPECTED_JSONL_SHA256}}")
    print(f"Verified expected JSONL: {{expected_path}}", flush=True)

    model_path = None
    for cand in sorted(input_base.rglob("*.pt")):
        if cand.is_file() and file_sha256(cand) == YOLO11L_SHA256:
            model_path = cand
            break

    # 2. Extract workspace source tree
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    b64_payload = "{b64_data}"
    archive_bytes = base64.b64decode(b64_payload)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        zf.extractall(work_root)
    print(f"Extracted workspace code into {{work_root}}", flush=True)

    # If model was not in inputs, check prepare.py
    if model_path is None:
        print("yolo11l.pt not found in attached inputs with matching hash, running prepare.py...", flush=True)
        run([sys.executable, "prepare.py"], cwd=work_root)
        local_model = work_root / "yolo11l.pt"
        if local_model.is_file() and file_sha256(local_model) == YOLO11L_SHA256:
            model_path = local_model
        else:
            raise FileNotFoundError(f"prepare.py did not produce yolo11l.pt with hash {{YOLO11L_SHA256}}")
    print(f"Verified model weights: {{model_path}}", flush=True)

    # 3. Install requirements
    print("Installing pinned requirements...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=work_root)

    # 4. Run local pytest suite
    print("Running test suite...", flush=True)
    run([sys.executable, "-m", "pytest", "-q"], cwd=work_root)

    # 5. Execute benchmark program
    artifacts_dir = working_base / "artifacts" / "batch_size"
    results_dir = working_base / "benchmarks" / "batch_size"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_cmd = [
        sys.executable,
        "scripts/benchmark_batch_size.py",
        "--fixture", str(fixture_path),
        "--expected", str(expected_path),
        "--model", str(model_path),
        "--device", "0",
        "--artifacts-dir", str(artifacts_dir),
        "--results-dir", str(results_dir),
    ]
    print("Launching benchmark driver...", flush=True)
    run(bench_cmd, cwd=work_root)

    # 6. Promote deliverables to /kaggle/working/
    for item in ["results.csv", "report.md", "throughput.png"]:
        src_item = results_dir / item
        if src_item.is_file():
            shutil.copy2(src_item, working_base / item)
            print(f"Promoted deliverable: {{working_base / item}}", flush=True)

    if (working_base / "report.md").is_file():
        print("\\n" + "=" * 60, flush=True)
        print("FINAL BENCHMARK REPORT:", flush=True)
        print("=" * 60, flush=True)
        print((working_base / "report.md").read_text(encoding="utf-8"), flush=True)

    print("Benchmark job finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

    out_file = ROOT / "kaggle" / "batch_size" / "benchmark.py"
    out_file.write_text(content, encoding="utf-8")
    print(f"Generated {out_file} ({len(content)} chars)")


if __name__ == "__main__":
    build_benchmark_script()
