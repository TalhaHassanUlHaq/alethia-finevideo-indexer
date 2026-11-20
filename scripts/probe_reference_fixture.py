from __future__ import annotations

import argparse
import io
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.althia.video import frame_timestamp

DEFAULT_FIXTURE_PATH = ROOT / "artifacts" / "reference" / "input" / "reference.parquet"
DEFAULT_PROBE_OUTPUT = ROOT / "artifacts" / "reference" / "probe.json"


def _format_fraction(value: Fraction | float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def probe_single_video(
    video_bytes: bytes,
    row_index: int,
    decoder: Callable[[bytes], Iterator[Any]] | None = None,
) -> dict[str, Any]:
    if not video_bytes:
        raise ValueError(f"row {row_index}: empty video payload")

    if decoder is not None:
        # Use custom/mock decoder for test environments
        frames = list(decoder(video_bytes))
        if not frames:
            raise RuntimeError(f"row {row_index}: decoder returned 0 frames")
        first_ts = frames[0].timestamp
        last_ts = frames[-1].timestamp
        frame_count = len(frames)
        elapsed = last_ts - first_ts
        if elapsed < 0:
            elapsed = Fraction(0)
        elapsed_final_second = int(elapsed.numerator // elapsed.denominator)
        img = frames[0].image
        decoded_dims = [int(img.shape[1]), int(img.shape[0])]
        rotations = list({getattr(f, "rotation", 0) for f in frames})
        return {
            "fixture_row": row_index,
            "frame_count": frame_count,
            "first_timestamp": _format_fraction(first_ts),
            "last_timestamp": _format_fraction(last_ts),
            "elapsed_final_second": elapsed_final_second,
            "stream_rate": 25.0,
            "encoded_dimensions": decoded_dims,
            "decoded_dimensions": decoded_dims,
            "rotation": rotations[0] if len(rotations) == 1 else rotations,
        }

    import av

    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"row {row_index}: container has no video stream")
        stream = container.streams.video[0]
        rate_val = stream.average_rate or stream.guessed_rate or stream.base_rate
        stream_rate = _format_fraction(rate_val) if rate_val else None

        encoded_dimensions = [int(stream.width), int(stream.height)]
        frame_count = 0
        first_ts: Fraction | None = None
        last_ts: Fraction | None = None
        decoded_dimensions: list[int] | None = None
        rotations: set[int] = set()

        for frame_index, frame in enumerate(container.decode(stream)):
            frame_count += 1
            ts = frame_timestamp(frame, frame_index, rate_val)
            if first_ts is None:
                first_ts = ts
            last_ts = ts

            rot = getattr(frame, "rotation", 0) or 0
            # Also check side data / stream metadata if frame.rotation is 0
            if rot == 0 and "rotate" in stream.metadata:
                try:
                    rot = int(stream.metadata["rotate"])
                except Exception:
                    pass
            rotations.add(int(rot))

            if decoded_dimensions is None:
                decoded_dimensions = [int(frame.width), int(frame.height)]

        if frame_count == 0 or first_ts is None or last_ts is None:
            raise RuntimeError(f"row {row_index}: failed to decode any frames")

        elapsed = last_ts - first_ts
        if elapsed < 0:
            elapsed = Fraction(0)
        elapsed_final_second = int(elapsed.numerator // elapsed.denominator)

        rot_list = sorted(rotations)
        return {
            "fixture_row": row_index,
            "frame_count": frame_count,
            "first_timestamp": _format_fraction(first_ts),
            "last_timestamp": _format_fraction(last_ts),
            "elapsed_final_second": elapsed_final_second,
            "stream_rate": stream_rate,
            "encoded_dimensions": encoded_dimensions,
            "decoded_dimensions": decoded_dimensions,
            "rotation": rot_list[0] if len(rot_list) == 1 else rot_list,
        }


def probe_fixture(
    fixture_path: Path,
    output_path: Path | None = None,
    decoder: Callable[[bytes], Iterator[Any]] | None = None,
) -> dict[str, Any]:
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture parquet file not found: {fixture_path}")

    parquet = pq.ParquetFile(fixture_path)
    video_records: list[dict[str, Any]] = []

    for row_index, batch in enumerate(parquet.iter_batches(batch_size=1, columns=["mp4"])):
        payload = batch.column(0)[0].as_py()
        info = probe_single_video(payload, row_index, decoder=decoder)
        video_records.append(info)

    if not video_records:
        raise ValueError(f"Fixture {fixture_path} contains zero videos")

    # Check whether at least one video has a nonzero rotation tag
    has_nonzero_rotation = False
    for v in video_records:
        rot = v["rotation"]
        if isinstance(rot, (int, float)) and rot != 0:
            has_nonzero_rotation = True
            break
        elif isinstance(rot, list) and any(r != 0 for r in rot):
            has_nonzero_rotation = True
            break

    warning = None
    if not has_nonzero_rotation:
        warning = "Acceptance warning: no video in reference fixture contains a confirmed nonzero rotation tag."

    probe_result = {
        "fixture_file": fixture_path.name,
        "video_count": len(video_records),
        "has_nonzero_rotation": has_nonzero_rotation,
        "warning": warning,
        "videos": video_records,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(probe_result, indent=2) + "\n", encoding="utf-8"
        )

    return probe_result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe reference fixture videos using PyAV."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Path to reference.parquet fixture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROBE_OUTPUT,
        help="Path to write probe JSON output.",
    )
    args = parser.parse_args()

    try:
        import av  # noqa: F401
    except ImportError:
        print(
            "PyAV is not installed locally. Local probe skipped as permitted by Phase 3 brief.",
            file=sys.stderr,
        )
        return 0

    try:
        result = probe_fixture(args.fixture, args.output)
        print(f"Probed {result['video_count']} videos successfully.")
        print(f"Nonzero rotation tag present: {result['has_nonzero_rotation']}")
        if result["warning"]:
            print(f"WARNING: {result['warning']}", file=sys.stderr)
        print(f"Probe output written to: {args.output}")
        return 0
    except Exception as exc:
        print(f"Probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
