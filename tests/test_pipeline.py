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


def test_default_cli_options_use_optimized_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.althia.cli import parse_args

    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "--input", "inputs", "--output", "results.jsonl"],
    )
    args = parse_args()
    assert args.batch_size == 16
    assert args.decoder_strategy == "D2"
    assert args.precision == "fp16"
    assert args.prefetch_capacity == 64


def test_cli_explicit_baseline_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.althia.cli import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--input",
            "inputs",
            "--output",
            "results.jsonl",
            "--decoder-strategy",
            "D0",
            "--precision",
            "fp32",
            "--batch-size",
            "1",
        ],
    )
    args = parse_args()
    assert args.batch_size == 1
    assert args.decoder_strategy == "D0"
    assert args.precision == "fp32"


def test_batch_size_validation_rejects_below_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        process_video("vid:0", b"test", FakeDetector(), batch_size=0)

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        process_video("vid:0", b"test", FakeDetector(), batch_size=-2)

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        run_pipeline(tmp_path, tmp_path / "out.jsonl", FakeDetector(), batch_size=0)


class RecordingDetector:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def detect_many(self, images: list[np.ndarray]) -> list[list[int]]:
        call_ids = [int(img[0, 0]) for img in images]
        self.calls.append(call_ids)
        # Class 0 (person) for even values, class 2 (car) for odd values
        return [[0] if val % 2 == 0 else [2] for val in call_ids]


def test_batched_frames_order_timestamps_and_pairing() -> None:
    # 7 frames across seconds 0, 1, 2
    # Second 0: 0.0 (val 0: person), 0.5 (val 1: car) -> ceil(2/2)=1 for each -> both person, car present, people=1 (median(1,0)=0.5 floored=0)
    # Second 1: 1.0 (val 2: person), 1.2 (val 3: car), 1.8 (val 4: person) -> 3 frames, person 2/3 (present), car 1/3 (not present)
    # Second 2: 2.0 (val 5: car), 2.5 (val 6: person) -> 2 frames, both present
    timestamps = [0.0, 0.5, 1.0, 1.2, 1.8, 2.0, 2.5]
    frames = [
        DecodedFrame(np.full((2, 2), i, dtype=np.uint8), Fraction(int(t * 10), 10))
        for i, t in enumerate(timestamps)
    ]

    detector = RecordingDetector()
    records, error = process_video(
        "test.parquet:0",
        b"data",
        detector,
        batch_size=3,
        decoder=lambda _: iter(frames),
    )

    assert error is None
    # 7 frames with batch_size 3: [0, 1, 2], [3, 4, 5], [6] (final incomplete batch)
    assert detector.calls == [[0, 1, 2], [3, 4, 5], [6]]

    # Ensure exact order and pairings carried through aggregation correctly
    assert len(records) == 3
    assert records[0]["second"] == 0
    assert records[0]["objects"] == ["person", "car"]
    assert records[1]["second"] == 1
    assert records[1]["objects"] == ["person"]
    assert records[2]["second"] == 2
    assert records[2]["objects"] == ["person", "car"]


def test_batches_never_cross_video_boundaries(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    shard = input_dir / "shard.parquet"
    make_parquet_file(shard, [b"vid0", b"vid1"])

    def two_video_decoder(payload: bytes):
        val = 0 if payload == b"vid0" else 10
        for i in range(3):
            yield DecodedFrame(np.full((1, 1), val + i, dtype=np.uint8), Fraction(i, 1))

    detector = RecordingDetector()
    output_path = tmp_path / "out.jsonl"

    import src.althia.pipeline as pipeline_mod
    orig_process = pipeline_mod.process_video

    def process_with_custom_decoder(*args, **kwargs):
        kwargs["decoder"] = two_video_decoder
        return orig_process(*args, **kwargs)

    try:
        pipeline_mod.process_video = process_with_custom_decoder
        run_pipeline(input_dir, output_path, detector, batch_size=4)
    finally:
        pipeline_mod.process_video = orig_process

    # With batch_size 4 and 3 frames per video:
    # Vid0 should be flushed at end of video as [0, 1, 2]
    # Vid1 should be flushed at end of video as [10, 11, 12]
    # A batch must NEVER cross video boundaries (no [0, 1, 2, 10] batch)
    assert detector.calls == [[0, 1, 2], [10, 11, 12]]


def test_batched_partial_failure_behavior() -> None:
    # 5 frames, then fail. batch_size = 4
    # Batch 1 will process [0, 1, 2, 3]
    # Frame 4 is in pending when decode fails partway, flushed on error
    def failing_decoder(_: bytes):
        for i in range(5):
            yield DecodedFrame(np.full((1, 1), i, dtype=np.uint8), Fraction(i, 1))
        raise RuntimeError("stream decode failed partway")

    detector = RecordingDetector()
    records, error = process_video(
        "fail.parquet:0",
        b"data",
        detector,
        batch_size=4,
        decoder=failing_decoder,
    )

    assert detector.calls == [[0, 1, 2, 3], [4]]
    assert len(records) == 5
    assert error == "RuntimeError: stream decode failed partway"

