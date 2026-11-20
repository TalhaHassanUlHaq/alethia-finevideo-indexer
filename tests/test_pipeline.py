from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.althia.pipeline import process_video, run_pipeline, short_error
from src.althia.validation import validate_jsonl
from src.althia.video import DecodedFrame


class FakeDetector:
    def detect_many(self, images: list[np.ndarray]) -> list[list[int]]:
        return [[0] if int(image[0, 0]) else [] for image in images]


def partial_decoder(_: bytes):
    yield DecodedFrame(np.ones((1, 1), dtype=np.uint8), Fraction(5, 1))
    yield DecodedFrame(np.zeros((1, 1), dtype=np.uint8), Fraction(11, 2))
    raise RuntimeError("damaged frame\nwith details")


def test_partial_failure_keeps_completed_frames_and_reports_error() -> None:
    records, error = process_video(
        "sample.parquet:4", b"video", FakeDetector(), batch_size=1, decoder=partial_decoder
    )
    assert records == [
        {"video": "sample.parquet:4", "second": 0, "objects": ["person"], "people": 0}
    ]
    assert error == "RuntimeError: damaged frame with details"


def test_empty_decode_becomes_error() -> None:
    records, error = process_video(
        "sample.parquet:0", b"video", FakeDetector(), decoder=lambda _: iter(())
    )
    assert records == []
    assert error == "could not decode video"


def test_error_messages_are_single_line_and_bounded() -> None:
    result = short_error(ValueError("x\ny" * 200))
    assert "\n" not in result
    assert len(result) == 200


def make_parquet_file(path: Path, payloads: list[bytes]) -> None:
    table = pa.Table.from_arrays([pa.array(payloads, type=pa.binary())], names=["mp4"])
    pq.write_table(table, path)


def test_pipeline_video_ids_fault_tolerance_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shard_0 = input_dir / "shard_000.parquet"
    shard_1 = input_dir / "shard_001.parquet"

    make_parquet_file(shard_0, [b"ok", b"fail_partway"])
    make_parquet_file(shard_1, [b"ok_after"])

    def mock_decoder(payload: bytes):
        if payload == b"ok":
            yield DecodedFrame(np.ones((1, 1), dtype=np.uint8), Fraction(0, 1))
            yield DecodedFrame(np.zeros((1, 1), dtype=np.uint8), Fraction(1, 1))
        elif payload == b"fail_partway":
            yield DecodedFrame(np.ones((1, 1), dtype=np.uint8), Fraction(0, 1))
            raise RuntimeError("stream decode failed partway")
        elif payload == b"ok_after":
            yield DecodedFrame(np.ones((1, 1), dtype=np.uint8), Fraction(0, 1))
        else:
            raise ValueError(f"unexpected payload: {payload!r}")

    monkeypatch.setattr(
        process_video,
        "__defaults__",
        (1, mock_decoder),
    )

    output_path = tmp_path / "results.jsonl"
    stats = run_pipeline(input_dir, output_path, FakeDetector(), batch_size=1)

    assert stats["videos"] == 3
    assert stats["errors"] == 1
    assert stats["seconds"] == 4

    lines = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert lines == [
        {"video": "shard_000.parquet:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "shard_000.parquet:0", "second": 1, "objects": [], "people": 0},
        {"video": "shard_000.parquet:1", "second": 0, "objects": ["person"], "people": 1},
        {"video": "shard_000.parquet:1", "error": "RuntimeError: stream decode failed partway"},
        {"video": "shard_001.parquet:0", "second": 0, "objects": ["person"], "people": 1},
    ]

    report = validate_jsonl(output_path, expected_videos=3, allow_errors=True)
    assert report == {"lines": 5, "videos": 3, "errors": 1}

    with pytest.raises(ValueError, match="found 1 error records"):
        validate_jsonl(output_path, expected_videos=3, allow_errors=False)


