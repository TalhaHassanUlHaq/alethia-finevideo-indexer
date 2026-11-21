from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_optimization_kernel() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in ["src", "scripts", "tests"]:
            for p in sorted((ROOT / folder).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    zf.write(p, p.relative_to(ROOT).as_posix())
        for single in ["run.py", "prepare.py", "requirements.txt"]:
            zf.write(ROOT / single, single)

    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

    kernel_dir = ROOT / "kaggle" / "optimization"
    kernel_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": "talhahassanulhaq/althia-optimization-benchmark",
        "title": "Althia Optimization Benchmark",
        "code_file": "optimize.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["talhahassanulhaq/althia-reference-suite"],
        "competition_sources": [],
        "kernel_sources": ["talhahassanulhaq/althia-phase-3-reference"],
        "model_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    script_content = f'''from __future__ import annotations

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
print("ALTHIA T4 COMPREHENSIVE OPTIMIZATION BENCHMARK RUNNER", flush=True)
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

    # 1. Locate authoritative input files by exact SHA-256
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

    # 2. Extract workspace archive
    work_root.mkdir(parents=True, exist_ok=True)
    zip_bytes = base64.b64decode("{b64_data}")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(work_root)
    print(f"Extracted workspace code into {{work_root}}", flush=True)

    # 3. Locate model weights
    model_path = None
    for cand in sorted(input_base.rglob("*.pt")):
        if cand.is_file() and file_sha256(cand) == YOLO11L_SHA256:
            model_path = cand
            break
    if model_path is None:
        raise FileNotFoundError(f"Could not locate yolo11l.pt matching hash {{YOLO11L_SHA256}}")
    local_model = work_root / "yolo11l.pt"
    if not local_model.is_file():
        shutil.copy2(model_path, local_model)
    print(f"Verified model weights: {{local_model}}", flush=True)

    # 4. Install pinned requirements
    print("Installing pinned requirements...", flush=True)
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=work_root)

    # 5. Run unit tests
    print("Running verification tests on Kaggle container...", flush=True)
    pytest_temp = working_base / "artifacts" / "pytest"
    pytest_temp.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(pytest_temp)], cwd=work_root)

    # 6. Record Environment Information
    import torch
    import pynvml
    import platform
    import psutil
    import json

    gpu_name = "Unknown"
    gpu_vram_mib = 0.0
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(h)
        gpu_vram_mib = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024)
    except Exception:
        pass

    import av
    import pyarrow
    import ultralytics
    import numpy

    env_info = {{
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "pyav_version": av.__version__,
        "pyarrow_version": pyarrow.__version__,
        "numpy_version": numpy.__version__,
        "gpu_name": gpu_name,
        "gpu_vram_mib": gpu_vram_mib,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gib": round(psutil.virtual_memory().total / (1024**3), 2),
    }}

    dest_benchmarks = working_base / "benchmarks" / "optimization"
    dest_artifacts = working_base / "artifacts" / "optimization"
    dest_benchmarks.mkdir(parents=True, exist_ok=True)
    dest_artifacts.mkdir(parents=True, exist_ok=True)

    env_path = dest_artifacts / "environment.json"
    env_path.write_text(json.dumps(env_info, indent=2) + "\\n", encoding="utf-8")
    print(f"Recorded environment info: {{env_path}}", flush=True)

    # 7. Run Decode Benchmark (Phase C)
    print("\\n" + "=" * 60, flush=True)
    print("RUNNING DECODE BENCHMARK (PHASE C)", flush=True)
    print("=" * 60, flush=True)
    decode_json = dest_artifacts / "decode_benchmark.json"
    run(
        [
            sys.executable,
            "scripts/benchmark_decoding.py",
            "--fixture",
            str(fixture_path),
            "--output",
            str(decode_json),
        ],
        cwd=work_root,
    )

    # 8. Run Comprehensive Optimization Benchmark in Driver Mode
    print("\\n" + "=" * 60, flush=True)
    print("RUNNING COMPREHENSIVE OPTIMIZATION MATRIX (PHASE D & E)", flush=True)
    print("=" * 60, flush=True)

    run(
        [
            sys.executable,
            "scripts/benchmark_configurations.py",
            "--driver",
            "--fixture",
            str(fixture_path),
            "--expected",
            str(expected_path),
            "--model",
            str(local_model),
            "--device",
            "0",
            "--output-dir",
            str(dest_benchmarks),
            "--artifacts-dir",
            str(dest_artifacts),
        ],
        cwd=work_root,
    )

    # 9. Promote root deliverables for easy download
    for name in [
        "configurations.csv",
        "raw_results.csv",
        "summary.csv",
        "report.md",
        "end_to_end_fps.png",
        "throughput_heatmap.png",
        "memory_vs_throughput.png",
        "decode_vs_inference.png",
        "jsonl_differences.png",
    ]:
        src_file = dest_benchmarks / name
        if src_file.is_file():
            shutil.copy2(src_file, working_base / name)
            print(f"Promoted deliverable: /kaggle/working/{{name}}", flush=True)

    print("\\n" + "=" * 60, flush=True)
    print("FINAL OPTIMIZATION REPORT:", flush=True)
    print("=" * 60, flush=True)
    report_file = dest_benchmarks / "report.md"
    if report_file.is_file():
        print(report_file.read_text(encoding="utf-8"), flush=True)

    print("\\nOptimization job finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    out_file = kernel_dir / "optimize.py"
    out_file.write_text(script_content, encoding="utf-8")
    print(f"Generated {out_file} ({len(script_content)} chars)")


if __name__ == "__main__":
    build_optimization_kernel()
