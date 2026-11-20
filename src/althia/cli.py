from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index FineVideo parquet files with YOLO11l."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("yolo11l.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise NotADirectoryError(args.input)
    from .inference import UltralyticsDetector

    detector = UltralyticsDetector(args.model, args.device)
    summary = run_pipeline(
        input_dir=args.input,
        output_path=args.output,
        detector=detector,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, sort_keys=True))
    # Per-video failures are represented in JSONL and must not abort a completed run.
    return 0

