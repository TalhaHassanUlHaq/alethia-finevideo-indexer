from __future__ import annotations

import hashlib
import io
import json
import sys
import types
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.build_reference_fixture import (
    build_reference_fixture,
    file_sha256,
    load_candidates,
)
from scripts.compare_reference import (
    THRESHOLD_OBJECT_F1,
    THRESHOLD_PEOPLE_EXACT,
    THRESHOLD_PEOPLE_WITHIN_ONE,
    compare_records,
)
from scripts.generate_reference import build_metadata, generate_reference
from scripts.probe_reference_fixture import probe_fixture
from src.althia.video import DecodedFrame


def create_synthetic_shard(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    mp4_data = [r.get("mp4", b"video_data") for r in rows]
    json_data = [
        {
            "duration_seconds": r.get("duration_seconds", 30),
            "resolution": r.get("resolution", "640x480"),
            "content_metadata": {"fps": r.get("fps", 25.0)},
        }
        for r in rows
    ]
    table = pa.Table.from_arrays(
        [
            pa.array(mp4_data, type=pa.binary()),
            pa.array(json_data),
        ],
        names=["mp4", "json"],
    )
    pq.write_table(table, path)


def test_load_candidates_valid_and_invalid(tmp_path: Path) -> None:
    valid_file = tmp_path / "candidates.json"
    valid_file.write_text(
        json.dumps([{"source_file": "s.parquet", "source_row": 0, "reason": "test"}]),
        encoding="utf-8",
    )
    loaded = load_candidates(valid_file)
    assert len(loaded) == 1
    assert loaded[0]["source_file"] == "s.parquet"

    with pytest.raises(FileNotFoundError):
        load_candidates(tmp_path / "nonexistent.json")

    empty_file = tmp_path / "empty.json"
    empty_file.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_candidates(empty_file)


def test_builder_deterministic_order_and_hashes(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_0 = shards_dir / "shard_0.parquet"
    shard_1 = shards_dir / "shard_1.parquet"

    create_synthetic_shard(
        shard_0,
        [
            {"mp4": b"s0_r0", "duration_seconds": 10},
            {"mp4": b"s0_r1", "duration_seconds": 20},
        ],
    )
    create_synthetic_shard(
        shard_1,
        [
            {"mp4": b"s1_r0", "duration_seconds": 30},
        ],
    )

    selection = [
        {"source_file": "shard_0.parquet", "source_row": 1, "reason": "first"},
        {"source_file": "shard_1.parquet", "source_row": 0, "reason": "second"},
        {"source_file": "shard_0.parquet", "source_row": 0, "reason": "third"},
    ]
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(json.dumps(selection), encoding="utf-8")

    out_dir = tmp_path / "out"
    fixture_parquet, manifest_path = build_reference_fixture(
        shards_dir=shards_dir,
        output_dir=out_dir,
        selection_path=sel_path,
    )

    assert fixture_parquet.is_file()
    assert manifest_path.is_file()

    # Verify row order in parquet
    pf = pq.ParquetFile(fixture_parquet)
    assert pf.metadata.num_rows == 3
    table = pf.read()
    payloads = table.column("mp4").to_pylist()
    assert payloads == [b"s0_r1", b"s1_r0", b"s0_r0"]

    # Verify manifest hashes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fixture_sha256"] == file_sha256(fixture_parquet)
    assert len(manifest["items"]) == 3
    for i, item in enumerate(manifest["items"]):
        assert item["fixture_row"] == i
        expected_digest = hashlib.sha256(payloads[i]).hexdigest()
        assert item["payload_sha256"] == expected_digest

    # Overwrite protection without --force when candidates change
    modified_sel = [{"source_file": "shard_0.parquet", "source_row": 0, "reason": "different"}]
    mod_path = tmp_path / "mod.json"
    mod_path.write_text(json.dumps(modified_sel), encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_reference_fixture(shards_dir, out_dir, mod_path, force=False)

    # With force=True, overwrite succeeds
    new_parquet, new_manifest = build_reference_fixture(shards_dir, out_dir, mod_path, force=True)
    assert pq.ParquetFile(new_parquet).metadata.num_rows == 1


def test_builder_streaming_does_not_call_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_0 = shards_dir / "shard_0.parquet"
    create_synthetic_shard(
        shard_0,
        [{"mp4": b"s0_r0"}, {"mp4": b"s0_r1"}, {"mp4": b"s0_r2"}],
    )

    selection = [{"source_file": "shard_0.parquet", "source_row": 1, "reason": "row1"}]
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(json.dumps(selection), encoding="utf-8")

    iter_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_iter = pq.ParquetFile.iter_batches

    def spy_iter_batches(self, *args, **kwargs):
        iter_calls.append((args, kwargs))
        return original_iter(self, *args, **kwargs)

    def fail_on_read(self, *args, **kwargs):
        raise AssertionError("ParquetFile.read() must not be called; bounded streaming iteration is required!")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy_iter_batches)
    monkeypatch.setattr(pq.ParquetFile, "read", fail_on_read)

    out_dir = tmp_path / "out"
    fixture_parquet, manifest_path = build_reference_fixture(
        shards_dir=shards_dir,
        output_dir=out_dir,
        selection_path=sel_path,
    )
    assert fixture_parquet.is_file()
    assert manifest_path.is_file()

    # Assert iter_batches was called with batch_size=1 and exactly mp4 and json columns
    assert len(iter_calls) == 1
    call_args, call_kwargs = iter_calls[0]
    batch_size = call_kwargs.get("batch_size") if "batch_size" in call_kwargs else (call_args[0] if call_args else None)
    columns = call_kwargs.get("columns") if "columns" in call_kwargs else (call_args[1] if len(call_args) > 1 else None)
    assert batch_size == 1
    assert columns == ["mp4", "json"]


def test_builder_missing_shard_fails(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(
        json.dumps([{"source_file": "missing.parquet", "source_row": 0}]),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        build_reference_fixture(shards_dir, tmp_path / "out", sel_path)


def test_builder_missing_row_fails(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard = shards_dir / "shard.parquet"
    create_synthetic_shard(shard, [{"mp4": b"row0"}])
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(
        json.dumps([{"source_file": "shard.parquet", "source_row": 99}]),
        encoding="utf-8",
    )
    with pytest.raises(IndexError, match="outside range"):
        build_reference_fixture(shards_dir, tmp_path / "out", sel_path)


def test_probe_fixture_with_mock_decoder(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.parquet"
    create_synthetic_shard(
        fixture_path,
        [
            {"mp4": b"vid_0"},
            {"mp4": b"vid_1"},
        ],
    )

    class RotatedFrame(DecodedFrame):
        rotation: int = 0

    def mock_decoder(payload: bytes):
        if payload == b"vid_0":
            f0 = RotatedFrame(np.zeros((100, 200, 3), dtype=np.uint8), Fraction(0, 1))
            f0.rotation = 0
            f1 = RotatedFrame(np.zeros((100, 200, 3), dtype=np.uint8), Fraction(1, 1))
            f1.rotation = 0
            yield f0
            yield f1
        elif payload == b"vid_1":
            f0 = RotatedFrame(np.zeros((200, 100, 3), dtype=np.uint8), Fraction(0, 1))
            f0.rotation = 90
            yield f0
        else:
            raise RuntimeError("corrupted video")

    out_probe = tmp_path / "probe.json"
    result = probe_fixture(fixture_path, out_probe, decoder=mock_decoder)

    assert result["video_count"] == 2
    assert result["has_nonzero_rotation"] is True
    assert result["warning"] is None
    assert len(result["videos"]) == 2
    assert result["videos"][0]["frame_count"] == 2
    assert result["videos"][1]["rotation"] == 90
    assert out_probe.is_file()

    # Test failure when decoder fails
    fail_fixture = tmp_path / "fail.parquet"
    create_synthetic_shard(fail_fixture, [{"mp4": b"broken"}])
    with pytest.raises(RuntimeError, match="corrupted"):
        probe_fixture(fail_fixture, decoder=mock_decoder)


def test_comparator_exact_equality(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    records = [
        {"video": "a:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "a:0", "second": 1, "objects": ["car", "dog"], "people": 0},
    ]
    content = "".join(json.dumps(r) + "\n" for r in records)
    ref_file.write_text(content, encoding="utf-8")
    cand_file.write_text(content, encoding="utf-8")

    metrics, passed = compare_records(ref_file, cand_file, exact=True)
    assert passed is True
    assert metrics["exact_record_match"]["rate"] == 1.0
    assert metrics["object_metrics"]["micro_f1"] == 1.0
    assert metrics["people_metrics"]["exact_rate"] == 1.0
    assert metrics["missing_records"] == 0
    assert metrics["unexpected_records"] == 0


def test_comparator_slight_differences_exact_vs_default(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    # 20 records
    ref_records = [
        {"video": "a:0", "second": i, "objects": ["person"], "people": 1}
        for i in range(20)
    ]
    cand_records = [
        {"video": "a:0", "second": i, "objects": ["person"], "people": 1}
        for i in range(20)
    ]
    # In record 0, candidate people is 2 instead of 1 (within one, but not exact)
    cand_records[0]["people"] = 2

    ref_file.write_text("".join(json.dumps(r) + "\n" for r in ref_records), encoding="utf-8")
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in cand_records), encoding="utf-8")

    # Default mode should pass (people exact = 19/20 = 95% >= 85%, within one = 100%, F1 = 1.0)
    metrics, passed_default = compare_records(ref_file, cand_file, exact=False)
    assert passed_default is True
    assert metrics["people_metrics"]["exact_rate"] == 0.95
    assert metrics["people_metrics"]["within_one_rate"] == 1.0
    assert metrics["object_metrics"]["micro_f1"] == 1.0

    # Exact mode should fail due to exact record mismatch
    metrics_exact, passed_exact = compare_records(ref_file, cand_file, exact=True)
    assert passed_exact is False
    assert metrics_exact["exact_record_match"]["rate"] == 0.95


def test_comparator_threshold_failures(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    # 10 records with 2 objects each
    ref_records = [
        {"video": "a:0", "second": i, "objects": ["person", "car"], "people": 2}
        for i in range(10)
    ]
    ref_file.write_text("".join(json.dumps(r) + "\n" for r in ref_records), encoding="utf-8")

    # Case A: F1 failure (candidate completely misses 'car', only detects 'person')
    cand_a = [
        {"video": "a:0", "second": i, "objects": ["person"], "people": 2}
        for i in range(10)
    ]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in cand_a), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=False)
    assert passed is False
    assert metrics["object_metrics"]["micro_f1"] < THRESHOLD_OBJECT_F1

    # Case B: People exact match failure (candidate differs in 5 out of 10 -> 50% < 85%)
    cand_b = [
        {"video": "a:0", "second": i, "objects": ["person", "car"], "people": 3 if i < 5 else 2}
        for i in range(10)
    ]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in cand_b), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=False)
    assert passed is False
    assert metrics["people_metrics"]["exact_rate"] < THRESHOLD_PEOPLE_EXACT
    assert metrics["people_metrics"]["within_one_rate"] >= THRESHOLD_PEOPLE_WITHIN_ONE

    # Case C: People within one failure (candidate differs by 5 in multiple records)
    cand_c = [
        {"video": "a:0", "second": i, "objects": ["person", "car"], "people": 10 if i < 3 else 2}
        for i in range(10)
    ]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in cand_c), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=False)
    assert passed is False
    assert metrics["people_metrics"]["within_one_rate"] < THRESHOLD_PEOPLE_WITHIN_ONE


def test_comparator_schema_and_missing_records(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "a:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "a:0", "second": 1, "objects": ["car"], "people": 0},
    ]
    ref_file.write_text("".join(json.dumps(r) + "\n" for r in ref_records), encoding="utf-8")

    # Missing second 1
    cand_missing = [{"video": "a:0", "second": 0, "objects": ["person"], "people": 1}]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in cand_missing), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=True)
    assert passed is False
    assert metrics["missing_records"] == 1

    # Schema failure (malformed line and invalid object name)
    cand_file.write_text(
        '{"video": "a:0", "second": 0, "objects": ["unknown_obj"], "people": 1}\nnot_json\n',
        encoding="utf-8",
    )
    metrics, passed = compare_records(ref_file, cand_file, exact=False)
    assert passed is False
    assert metrics["schema_failures"]["candidate_failures"] == 2


def test_comparator_exact_error_mismatches_and_ref_schema_failure(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref_err.jsonl"
    cand_file = tmp_path / "cand_err.jsonl"

    # Baseline identical mixed stream passes exact mode
    mixed_stream = [
        {"video": "a:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "a:0", "error": "RuntimeError: damaged frame"},
        {"video": "b:0", "second": 0, "objects": ["car"], "people": 0},
    ]
    content = "".join(json.dumps(r) + "\n" for r in mixed_stream)
    ref_file.write_text(content, encoding="utf-8")
    cand_file.write_text(content, encoding="utf-8")

    metrics, passed = compare_records(ref_file, cand_file, exact=True)
    assert passed is True
    assert metrics["error_records"]["exact_error_match"] is True

    # Case 1: Equal error count (1 error) but different error text
    diff_text = [
        {"video": "a:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "a:0", "error": "ValueError: different message"},
        {"video": "b:0", "second": 0, "objects": ["car"], "people": 0},
    ]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in diff_text), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=True)
    assert passed is False
    assert metrics["error_records"]["exact_error_match"] is False

    # Case 2: Equal error count (1 error) but different video ID
    diff_vid = [
        {"video": "a:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "b:0", "error": "RuntimeError: damaged frame"},
        {"video": "b:0", "second": 0, "objects": ["car"], "people": 0},
    ]
    cand_file.write_text("".join(json.dumps(r) + "\n" for r in diff_vid), encoding="utf-8")
    metrics, passed = compare_records(ref_file, cand_file, exact=True)
    assert passed is False
    assert metrics["error_records"]["exact_error_match"] is False

    # Case 3: Reference schema failure must fail both exact and threshold modes
    ref_bad = tmp_path / "ref_bad.jsonl"
    ref_bad.write_text("invalid_json_line\n", encoding="utf-8")
    metrics_exact, passed_exact = compare_records(ref_bad, cand_file, exact=True)
    assert passed_exact is False
    assert metrics_exact["schema_failures"]["reference_failures"] == 1

    metrics_thresh, passed_thresh = compare_records(ref_bad, cand_file, exact=False)
    assert passed_thresh is False
    assert metrics_thresh["schema_failures"]["reference_failures"] == 1


def test_generate_reference_enforces_all_fixture_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    shard_path = fixture_dir / "fixture.parquet"
    # Fixture with 2 videos
    create_synthetic_shard(shard_path, [{"mp4": b"v0"}, {"mp4": b"v1"}])

    fake_model = tmp_path / "yolo11l.pt"
    fake_model.write_text("fake_weights", encoding="utf-8")

    out_jsonl = tmp_path / "expected.jsonl"
    out_meta = tmp_path / "metadata.json"

    import types

    # Mock UltralyticsDetector without importing real ultralytics
    class FakeDetector:
        def __init__(self, model_path: Path, device: str = "auto") -> None:
            self.device = "cpu"

    fake_inference = types.ModuleType("src.althia.inference")
    fake_inference.UltralyticsDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "src.althia.inference", fake_inference)

    # Case 1: Pipeline only outputs 1 of the 2 expected videos -> must fail before writing metadata
    def mock_run_pipeline_missing(input_dir: Path, output_path: Path, detector: Any, batch_size: int = 1):
        output_path.write_text(
            json.dumps({"video": "fixture.parquet:0", "second": 0, "objects": [], "people": 0}) + "\n",
            encoding="utf-8",
        )
        return {"videos": 1, "errors": 0, "seconds": 1}

    monkeypatch.setattr("src.althia.pipeline.run_pipeline", mock_run_pipeline_missing)

    with pytest.raises(Exception):
        generate_reference(
            fixture_dir=fixture_dir,
            output_jsonl=out_jsonl,
            output_metadata=out_meta,
            model_path=fake_model,
        )
    assert not out_meta.exists()

    # Case 2: Pipeline outputs all 2 fixture videos -> must succeed and write metadata
    def mock_run_pipeline_complete(input_dir: Path, output_path: Path, detector: Any, batch_size: int = 1):
        output_path.write_text(
            json.dumps({"video": "fixture.parquet:0", "second": 0, "objects": [], "people": 0}) + "\n"
            + json.dumps({"video": "fixture.parquet:1", "second": 0, "objects": [], "people": 0}) + "\n",
            encoding="utf-8",
        )
        return {"videos": 2, "errors": 0, "seconds": 2}

    monkeypatch.setattr("src.althia.pipeline.run_pipeline", mock_run_pipeline_complete)

    generate_reference(
        fixture_dir=fixture_dir,
        output_jsonl=out_jsonl,
        output_metadata=out_meta,
        model_path=fake_model,
    )
    assert out_meta.exists()
    meta_data = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta_data["summary"]["videos"] == 2


def test_comparator_metrics_json_serialization_mixed_errors_and_empty_objects(tmp_path: Path) -> None:
    # 1. Identical mixed result/error streams
    ref_mixed = tmp_path / "ref_mixed.jsonl"
    cand_mixed = tmp_path / "cand_mixed.jsonl"

    stream = [
        {"video": "shard.parquet:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "shard.parquet:0", "error": "RuntimeError: decode failed"},
        {"video": "shard.parquet:1", "second": 0, "objects": ["car", "dog"], "people": 0},
    ]
    content = "".join(json.dumps(r) + "\n" for r in stream)
    ref_mixed.write_text(content, encoding="utf-8")
    cand_mixed.write_text(content, encoding="utf-8")

    metrics_mixed, passed_mixed = compare_records(ref_mixed, cand_mixed, exact=True)
    assert passed_mixed is True
    # Verify JSON serialization works without error
    json_str = json.dumps(metrics_mixed, indent=2)
    assert json_str is not None
    loaded = json.loads(json_str)
    assert isinstance(loaded["error_records"]["reference_error_counts"], list)
    assert loaded["error_records"]["reference_error_counts"] == [
        {"count": 1, "error": "RuntimeError: decode failed", "video": "shard.parquet:0"}
    ]

    # 2. Exactly equal all-empty-object stream
    ref_empty = tmp_path / "ref_empty.jsonl"
    cand_empty = tmp_path / "cand_empty.jsonl"

    empty_stream = [
        {"video": "shard.parquet:0", "second": 0, "objects": [], "people": 0},
        {"video": "shard.parquet:0", "second": 1, "objects": [], "people": 0},
        {"video": "shard.parquet:0", "second": 2, "objects": [], "people": 0},
    ]
    empty_content = "".join(json.dumps(r) + "\n" for r in empty_stream)
    ref_empty.write_text(empty_content, encoding="utf-8")
    cand_empty.write_text(empty_content, encoding="utf-8")

    metrics_empty, passed_empty = compare_records(ref_empty, cand_empty, exact=True)
    assert passed_empty is True
    assert metrics_empty["object_metrics"]["passed"] is True
    assert metrics_empty["object_metrics"]["micro_f1"] == 1.0
    assert metrics_empty["exact_record_match"]["rate"] == 1.0
    assert json.dumps(metrics_empty) is not None


def test_generate_reference_preserves_sentinels_on_incomplete_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    shard_path = fixture_dir / "fixture.parquet"
    create_synthetic_shard(shard_path, [{"mp4": b"v0"}, {"mp4": b"v1"}])

    fake_model = tmp_path / "yolo11l.pt"
    fake_model.write_text("fake_weights", encoding="utf-8")

    out_jsonl = tmp_path / "expected.jsonl"
    out_meta = tmp_path / "metadata.json"

    # Write sentinel authoritative files
    sentinel_jsonl_content = "SENTINEL_AUTHORITATIVE_JSONL\n"
    sentinel_meta_content = "SENTINEL_AUTHORITATIVE_META\n"
    out_jsonl.write_text(sentinel_jsonl_content, encoding="utf-8")
    out_meta.write_text(sentinel_meta_content, encoding="utf-8")

    class FakeDetector:
        def __init__(self, model_path: Path, device: str = "auto") -> None:
            self.device = "cpu"

    fake_inference = types.ModuleType("src.althia.inference")
    fake_inference.UltralyticsDetector = FakeDetector
    monkeypatch.setitem(sys.modules, "src.althia.inference", fake_inference)

    # Incomplete pipeline output: only 1 video out of 2
    def mock_run_pipeline_incomplete(input_dir: Path, output_path: Path, detector: Any, batch_size: int = 1):
        output_path.write_text(
            json.dumps({"video": "fixture.parquet:0", "second": 0, "objects": [], "people": 0}) + "\n",
            encoding="utf-8",
        )
        return {"videos": 1, "errors": 0, "seconds": 1}

    monkeypatch.setattr("src.althia.pipeline.run_pipeline", mock_run_pipeline_incomplete)

    with pytest.raises(Exception):
        generate_reference(
            fixture_dir=fixture_dir,
            output_jsonl=out_jsonl,
            output_metadata=out_meta,
            model_path=fake_model,
        )

    # Sentinels must remain byte-for-byte unchanged
    assert out_jsonl.read_text(encoding="utf-8") == sentinel_jsonl_content
    assert out_meta.read_text(encoding="utf-8") == sentinel_meta_content

    # No leftover temporary files
    leftover_tmps = [p.name for p in tmp_path.glob("*.tmp*")]
    assert leftover_tmps == []


def make_tiny_mp4() -> bytes:
    import av

    bio = io.BytesIO()
    with av.open(bio, mode="w", format="mp4") as container:
        stream = container.add_stream("h264", rate=25)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(5):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return bio.getvalue()


def test_builder_derived_rotation_transform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_0 = shards_dir / "shard_0.parquet"

    # Generate synthetic tiny MP4
    mp4_bytes = make_tiny_mp4()
    create_synthetic_shard(
        shard_0,
        [
            {"mp4": mp4_bytes, "duration_seconds": 5},
        ],
    )

    selection = [
        {"source_file": "shard_0.parquet", "source_row": 0, "reason": "original"},
        {
            "source_file": "shard_0.parquet",
            "source_row": 0,
            "transform": {"type": "rotate", "degrees": 90},
            "reason": "derived rotation 90",
        },
    ]
    sel_path = tmp_path / "sel.json"
    sel_path.write_text(json.dumps(selection), encoding="utf-8")

    iter_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    original_iter = pq.ParquetFile.iter_batches

    def spy_iter_batches(self, *args, **kwargs):
        iter_calls.append((args, kwargs))
        return original_iter(self, *args, **kwargs)

    def fail_on_read(self, *args, **kwargs):
        raise AssertionError("ParquetFile.read() must not be called!")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy_iter_batches)
    monkeypatch.setattr(pq.ParquetFile, "read", fail_on_read)

    out_dir_1 = tmp_path / "out_1"
    fix_1, man_1 = build_reference_fixture(
        shards_dir=shards_dir,
        output_dir=out_dir_1,
        selection_path=sel_path,
    )

    # Shard scanned only once despite duplicate source row
    assert len(iter_calls) == 1
    call_args, call_kwargs = iter_calls[0]
    batch_size = call_kwargs.get("batch_size") if "batch_size" in call_kwargs else (call_args[0] if call_args else None)
    assert batch_size == 1

    # Probe the fixture to test decoded rotation
    probe_res = probe_fixture(fix_1)
    assert probe_res["video_count"] == 2
    assert probe_res["has_nonzero_rotation"] is True
    assert probe_res["videos"][0]["rotation"] == 0
    assert probe_res["videos"][1]["rotation"] == 90
    assert probe_res["videos"][0]["frame_count"] == probe_res["videos"][1]["frame_count"] == 5
    assert probe_res["videos"][0]["encoded_dimensions"] == probe_res["videos"][1]["encoded_dimensions"] == [64, 64]

    # Check manifest details
    m1 = json.loads(man_1.read_text(encoding="utf-8"))
    assert m1["items"][0]["transform"] is None
    assert m1["items"][0]["payload_sha256"] == hashlib.sha256(mp4_bytes).hexdigest()
    assert m1["items"][0]["source_payload_sha256"] == hashlib.sha256(mp4_bytes).hexdigest()

    assert m1["items"][1]["transform"] == {"type": "rotate", "degrees": 90}
    assert m1["items"][1]["source_payload_sha256"] == hashlib.sha256(mp4_bytes).hexdigest()
    assert m1["items"][1]["payload_sha256"] != hashlib.sha256(mp4_bytes).hexdigest()

    # Second build must be identical (deterministic)
    out_dir_2 = tmp_path / "out_2"
    fix_2, man_2 = build_reference_fixture(
        shards_dir=shards_dir,
        output_dir=out_dir_2,
        selection_path=sel_path,
    )
    m2 = json.loads(man_2.read_text(encoding="utf-8"))
    assert m1["fixture_sha256"] == m2["fixture_sha256"]
    assert m1["items"][1]["payload_sha256"] == m2["items"][1]["payload_sha256"]


def test_metadata_provenance_duplicate_source_coordinates(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    fixture_parquet = fixture_dir / "reference.parquet"
    create_synthetic_shard(fixture_parquet, [{"mp4": b"v0"}, {"mp4": b"v1"}])

    fake_model = tmp_path / "yolo11l.pt"
    fake_model.write_text("fake_weights", encoding="utf-8")

    fake_results = tmp_path / "expected.jsonl"
    fake_results.write_text('{"video": "v0", "second": 0, "objects": [], "people": 0}\n', encoding="utf-8")

    # Manifest with duplicate source coordinates (e.g., row 0 control vs row 2 derived 90-degree transform)
    # Intentionally shuffle item ordering to test deterministic ordering by fixture_row
    manifest_items = [
        {
            "fixture_row": 2,
            "source_file": "shard_0.parquet",
            "source_row": 5,
            "source_payload_sha256": "orig_source_hash_row5",
            "payload_sha256": "derived_payload_hash_row5_rot90",
            "transform": {"type": "rotate", "degrees": 90},
        },
        {
            "fixture_row": 0,
            "source_file": "shard_0.parquet",
            "source_row": 5,
            "source_payload_sha256": "orig_source_hash_row5",
            "payload_sha256": "orig_source_hash_row5",
            "transform": None,
        },
        {
            "fixture_row": 1,
            "source_file": "shard_1.parquet",
            "source_row": 14,
            "source_payload_sha256": "orig_source_hash_s1_r14",
            "payload_sha256": "orig_source_hash_s1_r14",
            "transform": None,
        },
    ]
    manifest_path = tmp_path / "reference-manifest.json"
    manifest_path.write_text(json.dumps({"items": manifest_items}), encoding="utf-8")

    meta = build_metadata(
        fixture_dir=fixture_dir,
        output_jsonl=fake_results,
        model_path=fake_model,
        device="cpu",
        elapsed_seconds=12.34,
        pipeline_summary={"videos": 3, "errors": 0, "seconds": 75},
        manifest_path=manifest_path,
    )

    # 1. Unique original source payload hashes keyed by source coordinate without collision
    source_hashes = meta["hashes"]["source_payload_hashes"]
    assert len(source_hashes) == 2
    assert "shard_0.parquet:5" in source_hashes
    assert "shard_1.parquet:14" in source_hashes
    # Must preserve the original source payload hash, not the derived payload hash
    assert source_hashes["shard_0.parquet:5"] == "orig_source_hash_row5"
    assert source_hashes["shard_1.parquet:14"] == "orig_source_hash_s1_r14"

    # 2. Deterministic per-fixture-row list
    fixture_rows = meta["fixture_rows"]
    assert len(fixture_rows) == 3
    # Verify deterministic sorting by fixture_row (0, 1, 2)
    assert [r["fixture_row"] for r in fixture_rows] == [0, 1, 2]

    # Row 0 (control)
    r0 = fixture_rows[0]
    assert r0["fixture_row"] == 0
    assert r0["source_file"] == "shard_0.parquet"
    assert r0["source_row"] == 5
    assert r0["source_payload_sha256"] == "orig_source_hash_row5"
    assert r0["payload_sha256"] == "orig_source_hash_row5"
    assert r0["transform"] is None

    # Row 1
    r1 = fixture_rows[1]
    assert r1["fixture_row"] == 1
    assert r1["source_file"] == "shard_1.parquet"
    assert r1["source_row"] == 14
    assert r1["source_payload_sha256"] == "orig_source_hash_s1_r14"
    assert r1["payload_sha256"] == "orig_source_hash_s1_r14"
    assert r1["transform"] is None

    # Row 2 (derived transform sharing source coordinate with Row 0)
    r2 = fixture_rows[2]
    assert r2["fixture_row"] == 2
    assert r2["source_file"] == "shard_0.parquet"
    assert r2["source_row"] == 5
    assert r2["source_payload_sha256"] == "orig_source_hash_row5"
    assert r2["payload_sha256"] == "derived_payload_hash_row5_rot90"
    assert r2["transform"] == {"type": "rotate", "degrees": 90}

    # Proves Rows 0 and 2 have same source hash, different fixture payload hashes, and only Row 2 has transform
    assert r0["source_payload_sha256"] == r2["source_payload_sha256"]
    assert r0["payload_sha256"] != r2["payload_sha256"]

    # Also verify meta["hashes"]["fixture_rows"] matches
    assert meta["hashes"]["fixture_rows"] == fixture_rows

    # 3. JSON serializability
    serialized = json.dumps(meta, indent=2, sort_keys=True)
    deserialized = json.loads(serialized)
    assert deserialized["hashes"]["source_payload_hashes"]["shard_0.parquet:5"] == "orig_source_hash_row5"
    assert len(deserialized["fixture_rows"]) == 3

    # 4. Backward-compatible manifest without source_payload_sha256
    legacy_manifest_path = tmp_path / "legacy-manifest.json"
    legacy_manifest_path.write_text(
        json.dumps({
            "items": [
                {
                    "fixture_row": 0,
                    "source_file": "legacy.parquet",
                    "source_row": 1,
                    "payload_sha256": "legacy_payload_hash",
                }
            ]
        }),
        encoding="utf-8",
    )
    legacy_meta = build_metadata(
        fixture_dir=fixture_dir,
        output_jsonl=fake_results,
        model_path=fake_model,
        device="cpu",
        elapsed_seconds=1.0,
        pipeline_summary={"videos": 1, "errors": 0, "seconds": 10},
        manifest_path=legacy_manifest_path,
    )
    assert legacy_meta["hashes"]["source_payload_hashes"]["legacy.parquet:1"] == "legacy_payload_hash"
    assert legacy_meta["fixture_rows"][0]["source_payload_sha256"] == "legacy_payload_hash"



