from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
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
    parser.add_argument(
        "--decoder-strategy",
        choices=["D0", "D1", "D2", "D3"],
        default="D0",
        help="Video decoding strategy (default: D0)",
    )
    parser.add_argument(
        "--decode-thread-type",
        choices=["AUTO", "FRAME"],
        default="AUTO",
        help="PyAV thread type for D1/D3 (default: AUTO)",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=2,
        help="PyAV decode thread count for D1/D3 (default: 2)",
    )
    parser.add_argument(
        "--prefetch-capacity",
        type=int,
        default=64,
        help="Bounded queue capacity for D2/D3 prefetch (default: 64)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Model inference precision (default: fp32)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise NotADirectoryError(args.input)
    from .inference import UltralyticsDetector

    config = PipelineConfig(
        decoder_strategy=args.decoder_strategy,
        decode_thread_type=args.decode_thread_type,
        decode_threads=args.decode_threads,
        prefetch_capacity=args.prefetch_capacity,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    detector = UltralyticsDetector(
        model_path=args.model,
        device=args.device,
        precision=config.precision,
    )
    summary = run_pipeline(
        input_dir=args.input,
        output_path=args.output,
        detector=detector,
        config=config,
    )
    print(json.dumps(summary, sort_keys=True))
    # Per-video failures are represented in JSONL and must not abort a completed run.
    return 0
