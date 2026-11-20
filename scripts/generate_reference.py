from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FIXTURE_DIR = ROOT / "artifacts" / "reference" / "input"
DEFAULT_OUTPUT_JSONL = ROOT / "reference" / "expected.jsonl"
DEFAULT_OUTPUT_METADATA = ROOT / "reference" / "metadata.json"
DEFAULT_MODEL = ROOT / "yolo11l.pt"
DEFAULT_MANIFEST = ROOT / "artifacts" / "reference" / "input" / "reference-manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("torch", "torchvision", "ultralytics", "av", "pyarrow", "numpy"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"
    return versions


def build_metadata(
    fixture_dir: Path,
    output_jsonl: Path,
    model_path: Path,
    device: str,
    elapsed_seconds: float,
    pipeline_summary: dict[str, Any],
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    fixture_files = sorted(fixture_dir.glob("*.parquet"))
    fixture_hashes = {f.name: file_sha256(f) for f in fixture_files}

    source_hashes: dict[str, str] = {}
    fixture_rows: list[dict[str, Any]] = []
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            items = manifest_data.get("items", [])
            sorted_items = sorted(
                items,
                key=lambda it: (it.get("fixture_row") is None, it.get("fixture_row", 0)),
            )
            for item in sorted_items:
                source_file = item.get("source_file")
                source_row = item.get("source_row")
                src_hash = (
                    item.get("source_payload_sha256")
                    or item.get("payload_sha256")
                    or ""
                )
                fixture_hash = item.get("payload_sha256", "")
                transform = item.get("transform", None)
                if source_file is not None and source_row is not None:
                    coord = f"{source_file}:{source_row}"
                    if coord not in source_hashes or item.get("source_payload_sha256"):
                        source_hashes[coord] = src_hash

                fixture_rows.append(
                    {
                        "fixture_row": item.get("fixture_row"),
                        "source_file": source_file,
                        "source_row": source_row,
                        "source_payload_sha256": src_hash,
                        "payload_sha256": fixture_hash,
                        "fixture_payload_sha256": fixture_hash,
                        "transform": transform,
                    }
                )
            source_hashes = dict(sorted(source_hashes.items()))
        except Exception:
            pass

    return {
        "generator": "scripts/generate_reference.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "model_settings": {
            "model_file": model_path.name,
            "imgsz": 640,
            "conf": 0.4,
            "iou": 0.7,
            "precision": "FP32",
            "batch_size": 1,
        },
        "hashes": {
            "model_weight_sha256": file_sha256(model_path) if model_path.is_file() else None,
            "result_sha256": file_sha256(output_jsonl) if output_jsonl.is_file() else None,
            "fixture_hashes": fixture_hashes,
            "source_payload_hashes": source_hashes,
            "fixture_rows": fixture_rows,
        },
        "fixture_rows": fixture_rows,
        "packages": get_package_versions(),
        "summary": pipeline_summary,
    }


def generate_reference(
    fixture_dir: Path,
    output_jsonl: Path,
    output_metadata: Path,
    model_path: Path = DEFAULT_MODEL,
    device: str = "auto",
    manifest_path: Path = DEFAULT_MANIFEST,
) -> None:
    if not fixture_dir.is_dir():
        raise NotADirectoryError(f"Fixture directory not found: {fixture_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {model_path}. Run 'python prepare.py' first."
        )

    parquet_files = sorted(fixture_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in fixture directory: {fixture_dir}")

    # Derive expected video count directly from input Parquet metadata
    expected_videos = sum(
        pq.ParquetFile(p).metadata.num_rows for p in parquet_files
    )
    if expected_videos <= 0:
        raise ValueError(f"Fixture directory {fixture_dir} contains zero video rows across parquet files")

    from src.althia.inference import UltralyticsDetector
    from src.althia.pipeline import run_pipeline
    from src.althia.validation import validate_jsonl

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    tmp_suffix = f".tmp_{os.getpid()}_{time.time_ns()}"
    tmp_jsonl = output_jsonl.parent / f"{output_jsonl.name}{tmp_suffix}"
    tmp_metadata = output_metadata.parent / f"{output_metadata.name}{tmp_suffix}"

    try:
        detector = UltralyticsDetector(model_path=model_path, device=device)
        actual_device = detector.device

        print(
            f"Running reference pipeline on {fixture_dir} with device={actual_device} "
            f"(expecting {expected_videos} videos)..."
        )
        started = time.perf_counter()
        summary = run_pipeline(
            input_dir=fixture_dir,
            output_path=tmp_jsonl,
            detector=detector,
            batch_size=1,
        )
        elapsed = time.perf_counter() - started

        # Verify pipeline summary processed all fixture videos without error records
        if summary.get("videos") != expected_videos:
            raise RuntimeError(
                f"Pipeline summary reported {summary.get('videos')} videos, but fixture requires {expected_videos}"
            )
        if summary.get("errors", 0) != 0:
            raise RuntimeError(
                f"Pipeline summary reported {summary.get('errors')} error records during reference generation"
            )

        # Validate output results enforcing all fixture videos and zero errors
        validation_report = validate_jsonl(
            tmp_jsonl, expected_videos=expected_videos, allow_errors=False
        )
        print(f"Validated reference output: {validation_report}")

        metadata = build_metadata(
            fixture_dir=fixture_dir,
            output_jsonl=tmp_jsonl,
            model_path=model_path,
            device=actual_device,
            elapsed_seconds=elapsed,
            pipeline_summary=summary,
            manifest_path=manifest_path,
        )
        tmp_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Atomically replace authoritative outputs once fully validated
        os.replace(tmp_jsonl, output_jsonl)
        os.replace(tmp_metadata, output_metadata)

        print(f"Reference answers written to: {output_jsonl}")
        print(f"Reference metadata written to: {output_metadata}")
    finally:
        if tmp_jsonl.exists():
            try:
                tmp_jsonl.unlink()
            except OSError:
                pass
        if tmp_metadata.exists():
            try:
                tmp_metadata.unlink()
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate stable reference answers using baseline FP32 inference."
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=DEFAULT_FIXTURE_DIR,
        help="Directory containing reference parquet fixture.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Path for generated reference expected.jsonl.",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=DEFAULT_OUTPUT_METADATA,
        help="Path for generated reference metadata.json.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to YOLO11l weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use ('auto', '0', 'cpu', etc.).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to reference fixture manifest.",
    )
    args = parser.parse_args()

    try:
        generate_reference(
            fixture_dir=args.fixture_dir,
            output_jsonl=args.output_jsonl,
            output_metadata=args.output_metadata,
            model_path=args.model,
            device=args.device,
            manifest_path=args.manifest,
        )
        return 0
    except Exception as exc:
        print(f"Reference generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
