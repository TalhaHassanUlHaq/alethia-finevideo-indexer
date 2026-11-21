import json
from pathlib import Path

from scripts.compare_predictions import compare_jsonl_predictions


def test_compare_jsonl_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person", "car"], "people": 1},
        {"video": "v1.mp4:0", "second": 1, "objects": ["dog"], "people": 0},
    ]

    for p in (ref_file, cand_file):
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is True
    assert res["differing_records_count"] == 0
    assert res["object_micro_f1"] == 1.0
    assert res["people_exact_pct"] == 100.0
    assert res["schema_violations_count"] == 0
    assert res["duplicate_keys_count"] == 0
    assert res["parse_errors_count"] == 0


def test_compare_jsonl_fp16_style_difference(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person", "car"], "people": 2},
        {"video": "v1.mp4:0", "second": 1, "objects": ["dog"], "people": 0},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},  # missing car, people diff by 1
        {"video": "v1.mp4:0", "second": 1, "objects": ["dog"], "people": 0},
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["differing_records_count"] == 1
    assert res["people_exact_matches"] == 1
    assert res["people_within_one_matches"] == 2
    assert res["people_within_one_pct"] == 100.0
    assert res["actual_assignment_accuracy"] == "unverified (no ground truth available)"


def test_compare_jsonl_malformed_json_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
    ]
    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    # Candidate has a broken JSON line
    cand_file.write_text('{"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1\n', encoding="utf-8")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["parse_errors_count"] == 1


def test_compare_jsonl_schema_invalid_keys_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"]},  # missing 'people' key
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["schema_violations_count"] == 1


def test_compare_jsonl_duplicate_keys_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},  # duplicate
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["duplicate_keys_count"] == 1


def test_compare_jsonl_unknown_class_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person", "airplane"], "people": 1},
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["unknown_class_violations"] == 1


def test_compare_jsonl_non_canonical_order_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person", "car"], "people": 1},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["car", "person"], "people": 1},  # reversed order
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["canonical_order_violations"] == 1


def test_compare_jsonl_differing_error_records_fails_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    ref_records = [
        {"video": "v1.mp4:0", "error": "corrupt video stream"},
    ]
    cand_records = [
        {"video": "v1.mp4:0", "error": "different error message"},
    ]

    with ref_file.open("w", encoding="utf-8") as f:
        for r in ref_records:
            f.write(json.dumps(r) + "\n")

    with cand_file.open("w", encoding="utf-8") as f:
        for r in cand_records:
            f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is False
    assert res["error_record_mismatches"] == 1


def test_compare_jsonl_matching_error_records_passes_exact_match(tmp_path: Path) -> None:
    ref_file = tmp_path / "ref.jsonl"
    cand_file = tmp_path / "cand.jsonl"

    records = [
        {"video": "v1.mp4:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "v2.mp4:0", "error": "corrupt video stream"},
    ]

    for p in (ref_file, cand_file):
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    res = compare_jsonl_predictions(cand_file, ref_file)
    assert res["exact_match"] is True
    assert res["error_record_mismatches"] == 0
