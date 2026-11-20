import json
from pathlib import Path

import pytest

from src.althia.validation import validate_jsonl


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_valid_output(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_records(
        path,
        [
            {"video": "a:0", "second": 0, "objects": [], "people": 0},
            {"video": "a:0", "second": 1, "objects": ["person", "cell phone"], "people": 1},
        ],
    )
    assert validate_jsonl(path, expected_videos=1, allow_errors=False) == {
        "lines": 2,
        "videos": 1,
        "errors": 0,
    }


def test_rejects_gaps_and_noncanonical_objects(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_records(
        path,
        [{"video": "a:0", "second": 1, "objects": ["dog", "person"], "people": 0}],
    )
    with pytest.raises(ValueError):
        validate_jsonl(path)


def test_validate_jsonl_mixed_results_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_records(
        path,
        [
            {"video": "shard_a.parquet:0", "second": 0, "objects": ["person"], "people": 1},
            {"video": "shard_a.parquet:0", "second": 1, "objects": [], "people": 0},
            {"video": "shard_a.parquet:1", "second": 0, "objects": ["person"], "people": 1},
            {"video": "shard_a.parquet:1", "error": "RuntimeError: stream decode failed"},
            {"video": "shard_b.parquet:0", "second": 0, "objects": ["car"], "people": 0},
        ],
    )
    report = validate_jsonl(path, expected_videos=3, allow_errors=True)
    assert report == {"lines": 5, "videos": 3, "errors": 1}

    with pytest.raises(ValueError, match="found 1 error records"):
        validate_jsonl(path, allow_errors=False)

