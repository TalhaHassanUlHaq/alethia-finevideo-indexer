from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Protocol, Sequence

import numpy as np
import pyarrow.parquet as pq

from .aggregation import VideoAggregation
from .config import PipelineConfig
from .video import DecodedFrame, decode_video, get_decoder_for_config


class Detector(Protocol):
    def detect_many(self, images: Sequence[np.ndarray]) -> list[list[int]]: ...


def short_error(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    return text[:200]


def process_video(
    video_id: str,
    video_bytes: bytes,
    detector: Detector,
    batch_size: int = 1,
    decoder: Callable[[bytes], Iterable[DecodedFrame]] = decode_video,
) -> tuple[list[dict[str, object]], str | None]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    aggregation = VideoAggregation()
    pending: list[DecodedFrame] = []
    error: str | None = None

    def flush_pending() -> None:
        if not pending:
            return
        detections = detector.detect_many([frame.image for frame in pending])
        for frame, class_ids in zip(pending, detections, strict=True):
            aggregation.add_frame(frame.timestamp, class_ids)
        pending.clear()

    try:
        for frame in decoder(video_bytes):
            pending.append(frame)
            if len(pending) >= batch_size:
                flush_pending()
    except Exception as exc:  # Per-video isolation is an assignment requirement.
        error = short_error(exc)

    try:
        flush_pending()
    except Exception as exc:
        error = short_error(exc)

    records = aggregation.records(video_id)
    if not records and error is None:
        error = "could not decode video"
    return records, error


def iter_parquet_videos(path: Path) -> Iterator[tuple[int, bytes]]:
    parquet = pq.ParquetFile(path)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=1, columns=["mp4"]):
        payload = batch.column(0)[0].as_py()
        yield row_number, payload
        row_number += 1


def write_json_line(stream: object, record: dict[str, object]) -> None:
    stream.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
    stream.write("\n")


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    detector: Detector,
    batch_size: int = 1,
    decoder: Callable[[bytes], Iterable[DecodedFrame]] | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, float | int]:
    if config is not None:
        effective_batch_size = config.batch_size
        effective_decoder = (
            get_decoder_for_config(config) if decoder is None else decoder
        )
        custom_decoder = decoder is not None or config.decoder_strategy != "D0"
    else:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        effective_batch_size = batch_size
        effective_decoder = decode_video if decoder is None else decoder
        custom_decoder = decoder is not None

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no .parquet files found in {input_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    videos = 0
    errors = 0
    seconds = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for parquet_path in parquet_files:
            try:
                for row_number, payload in iter_parquet_videos(parquet_path):
                    video_id = f"{parquet_path.name}:{row_number}"
                    if custom_decoder:
                        records, error = process_video(
                            video_id=video_id,
                            video_bytes=payload,
                            detector=detector,
                            batch_size=effective_batch_size,
                            decoder=effective_decoder,
                        )
                    else:
                        records, error = process_video(
                            video_id=video_id,
                            video_bytes=payload,
                            detector=detector,
                            batch_size=effective_batch_size,
                        )
                    for record in records:
                        write_json_line(output, record)
                    if error is not None:
                        write_json_line(output, {"video": video_id, "error": error})
                        errors += 1
                    output.flush()
                    videos += 1
                    seconds += len(records)
                    print(
                        f"processed {video_id}: {len(records)} seconds"
                        + (f"; {error}" if error else ""),
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                errors += 1
                print(
                    f"failed to read {parquet_path.name}: {short_error(exc)}",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "videos": videos,
        "errors": errors,
        "seconds": seconds,
        "elapsed_seconds": time.perf_counter() - started,
    }
