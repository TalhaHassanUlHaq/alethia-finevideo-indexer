from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.althia.constants import TARGET_OBJECTS

TARGET_CLASSES = list(TARGET_OBJECTS)
CANONICAL_INDEX = {name: i for i, name in enumerate(TARGET_CLASSES)}


def load_jsonl_records_with_diagnostics(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load JSONL lines safely while capturing parse errors line-by-line."""
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []

    if not path.is_file():
        parse_errors.append({"line": 0, "error": f"File not found: {path}"})
        return records, parse_errors

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    data["_line_no"] = line_no
                    records.append(data)
                else:
                    parse_errors.append(
                        {"line": line_no, "error": "Record is not a JSON object"}
                    )
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_no, "error": str(exc)})

    return records, parse_errors


def compare_jsonl_predictions(
    candidate_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    cand_records, cand_parse_errors = load_jsonl_records_with_diagnostics(candidate_path)
    ref_records, ref_parse_errors = load_jsonl_records_with_diagnostics(reference_path)

    # 1. Parse and validate reference records
    ref_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    ref_errors: list[dict[str, Any]] = []
    for r in ref_records:
        if "error" in r:
            ref_errors.append(r)
        else:
            ref_by_key[(str(r.get("video")), int(r.get("second", -1)))] = r

    # 2. Parse and validate candidate records with schema and duplication checks
    cand_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    cand_errors: list[dict[str, Any]] = []
    schema_violations: list[dict[str, Any]] = []
    duplicate_keys: list[dict[str, Any]] = []
    canonical_order_violations = 0
    unknown_class_violations = 0

    seen_cand_keys: set[tuple[str, int]] = set()

    for r in cand_records:
        line_no = r.get("_line_no", -1)
        video = r.get("video")

        if not isinstance(video, str) or not video:
            schema_violations.append(
                {"line": line_no, "error": "Missing or invalid 'video' field"}
            )
            continue

        if "error" in r:
            clean_keys = set(r.keys()) - {"_line_no"}
            if clean_keys != {"video", "error"} or not isinstance(r["error"], str):
                schema_violations.append(
                    {"line": line_no, "error": "Invalid error record schema"}
                )
            cand_errors.append(r)
            continue

        clean_keys = set(r.keys()) - {"_line_no"}
        if clean_keys != {"video", "second", "objects", "people"}:
            schema_violations.append(
                {
                    "line": line_no,
                    "error": f"Invalid result keys: {sorted(list(clean_keys))}",
                }
            )
            continue

        second = r.get("second")
        people = r.get("people")
        objects = r.get("objects")

        if not isinstance(second, int) or isinstance(second, bool) or second < 0:
            schema_violations.append(
                {"line": line_no, "error": f"Invalid second: {second}"}
            )
            continue

        if not isinstance(people, int) or isinstance(people, bool) or people < 0:
            schema_violations.append(
                {"line": line_no, "error": f"Invalid people count: {people}"}
            )
            continue

        if not isinstance(objects, list):
            schema_violations.append(
                {"line": line_no, "error": "objects field must be a list"}
            )
            continue

        # Check duplicate keys
        key = (video, second)
        if key in seen_cand_keys:
            duplicate_keys.append(
                {"line": line_no, "video": video, "second": second}
            )
        else:
            seen_cand_keys.add(key)
            cand_by_key[key] = r

        # Check unknown classes and duplicates in objects
        has_unknown = False
        for obj in objects:
            if not isinstance(obj, str) or obj not in CANONICAL_INDEX:
                unknown_class_violations += 1
                has_unknown = True
        if len(set(objects)) != len(objects):
            schema_violations.append(
                {"line": line_no, "error": "Duplicate objects in list"}
            )

        # Check canonical ordering
        if not has_unknown:
            expected_order = sorted(objects, key=CANONICAL_INDEX.__getitem__)
            if objects != expected_order:
                canonical_order_violations += 1

    # 3. Compare error records
    error_record_mismatches = 0
    ref_err_map = {str(r.get("video")): str(r.get("error")) for r in ref_errors}
    cand_err_map = {str(r.get("video")): str(r.get("error")) for r in cand_errors}
    if ref_err_map != cand_err_map:
        error_record_mismatches = max(len(ref_err_map), len(cand_err_map)) - len(
            set(ref_err_map.items()) & set(cand_err_map.items())
        )

    # 4. Key difference analysis
    ref_keys = set(ref_by_key.keys())
    cand_keys = set(cand_by_key.keys())

    missing_keys = sorted(list(ref_keys - cand_keys))
    unexpected_keys = sorted(list(cand_keys - ref_keys))
    common_keys = sorted(list(ref_keys & cand_keys))

    differing_records_count = len(missing_keys) + len(unexpected_keys)
    per_class_tp = {cls: 0 for cls in TARGET_CLASSES}
    per_class_fp = {cls: 0 for cls in TARGET_CLASSES}
    per_class_fn = {cls: 0 for cls in TARGET_CLASSES}

    people_exact_matches = 0
    people_within_one_matches = 0
    object_exact_matches = 0

    detailed_mismatches: list[dict[str, Any]] = []

    for key in common_keys:
        ref_rec = ref_by_key[key]
        cand_rec = cand_by_key[key]

        ref_objs = ref_rec.get("objects", [])
        cand_objs = cand_rec.get("objects", [])

        ref_set = set(ref_objs)
        cand_set = set(cand_objs)

        for cls in TARGET_CLASSES:
            in_ref = cls in ref_set
            in_cand = cls in cand_set
            if in_ref and in_cand:
                per_class_tp[cls] += 1
            elif not in_ref and in_cand:
                per_class_fp[cls] += 1
            elif in_ref and not in_cand:
                per_class_fn[cls] += 1

        ref_people = int(ref_rec.get("people", 0))
        cand_people = int(cand_rec.get("people", 0))

        is_obj_match = (ref_objs == cand_objs)
        if is_obj_match:
            object_exact_matches += 1

        is_people_exact = (ref_people == cand_people)
        if is_people_exact:
            people_exact_matches += 1

        is_people_within_one = abs(ref_people - cand_people) <= 1
        if is_people_within_one:
            people_within_one_matches += 1

        if not (is_obj_match and is_people_exact):
            differing_records_count += 1
            detailed_mismatches.append(
                {
                    "video": key[0],
                    "second": key[1],
                    "ref_objects": ref_objs,
                    "cand_objects": cand_objs,
                    "ref_people": ref_people,
                    "cand_people": cand_people,
                }
            )

    total_evaluated = len(ref_keys)

    # Strict exact match definition: MUST have zero differing records, zero missing/unexpected,
    # zero parse errors, zero schema violations, zero duplicate keys, zero canonical violations,
    # zero unknown classes, and zero error mismatches.
    exact_match = (
        differing_records_count == 0
        and len(missing_keys) == 0
        and len(unexpected_keys) == 0
        and len(cand_parse_errors) == 0
        and len(ref_parse_errors) == 0
        and len(schema_violations) == 0
        and len(duplicate_keys) == 0
        and canonical_order_violations == 0
        and unknown_class_violations == 0
        and error_record_mismatches == 0
        and len(cand_errors) == len(ref_errors)
    )

    # Micro F1 computation
    total_tp = sum(per_class_tp.values())
    total_fp = sum(per_class_fp.values())
    total_fn = sum(per_class_fn.values())

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = (
        2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 1.0
    )

    exact_record_pct = (
        (total_evaluated - differing_records_count) / total_evaluated * 100.0
        if total_evaluated > 0
        else 0.0
    )
    people_exact_pct = (
        people_exact_matches / len(common_keys) * 100.0 if common_keys else 0.0
    )
    people_within_one_pct = (
        people_within_one_matches / len(common_keys) * 100.0 if common_keys else 0.0
    )

    meets_brief_thresholds_vs_ref = (
        micro_f1 >= 0.95
        and people_exact_pct >= 85.0
        and people_within_one_pct >= 95.0
    )

    return {
        "candidate_file": str(candidate_path),
        "reference_file": str(reference_path),
        "exact_match": exact_match,
        "total_reference_records": total_evaluated,
        "total_candidate_records": len(cand_records),
        "differing_records_count": differing_records_count,
        "exact_record_pct": round(exact_record_pct, 2),
        "missing_records_count": len(missing_keys),
        "unexpected_records_count": len(unexpected_keys),
        "parse_errors_count": len(cand_parse_errors),
        "schema_violations_count": len(schema_violations),
        "duplicate_keys_count": len(duplicate_keys),
        "canonical_order_violations": canonical_order_violations,
        "unknown_class_violations": unknown_class_violations,
        "error_record_mismatches": error_record_mismatches,
        "object_micro_precision": round(micro_prec, 4),
        "object_micro_recall": round(micro_rec, 4),
        "object_micro_f1": round(micro_f1, 4),
        "people_exact_matches": people_exact_matches,
        "people_exact_pct": round(people_exact_pct, 2),
        "people_within_one_matches": people_within_one_matches,
        "people_within_one_pct": round(people_within_one_pct, 2),
        "meets_brief_thresholds_vs_ref": meets_brief_thresholds_vs_ref,
        "actual_assignment_accuracy": "unverified (no ground truth available)",
        "per_class_tp": per_class_tp,
        "per_class_fp": per_class_fp,
        "per_class_fn": per_class_fn,
        "sample_mismatches": detailed_mismatches[:50],
        "diagnostics": {
            "cand_parse_errors": cand_parse_errors[:10],
            "schema_violations": schema_violations[:10],
            "duplicate_keys": duplicate_keys[:10],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare JSONL predictions against reference"
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    report = compare_jsonl_predictions(args.candidate, args.reference)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    print(f"Exact match: {report['exact_match']}")
    print(f"Object Micro F1: {report['object_micro_f1']}")
    print(
        f"People exact: {report['people_exact_pct']}% | within-one: {report['people_within_one_pct']}%"
    )
    print(f"Differing records: {report['differing_records_count']}")
    if report["parse_errors_count"] > 0:
        print(f"Parse errors: {report['parse_errors_count']}")
    if report["schema_violations_count"] > 0:
        print(f"Schema violations: {report['schema_violations_count']}")
    if report["duplicate_keys_count"] > 0:
        print(f"Duplicate keys: {report['duplicate_keys_count']}")
    if report["canonical_order_violations"] > 0:
        print(f"Canonical order violations: {report['canonical_order_violations']}")
    if report["unknown_class_violations"] > 0:
        print(f"Unknown class violations: {report['unknown_class_violations']}")
    if report["error_record_mismatches"] > 0:
        print(f"Error record mismatches: {report['error_record_mismatches']}")

    return 0 if report["exact_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
