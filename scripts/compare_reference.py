from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.althia.constants import TARGET_OBJECTS

DEFAULT_REFERENCE = ROOT / "reference" / "expected.jsonl"
CANONICAL_INDEX = {name: index for index, name in enumerate(TARGET_OBJECTS)}

THRESHOLD_OBJECT_F1 = 0.95
THRESHOLD_PEOPLE_EXACT = 0.85
THRESHOLD_PEOPLE_WITHIN_ONE = 0.95


def serialize_error_counts(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"video": video, "error": error_msg, "count": count}
        for (video, error_msg), count in sorted(counter.items())
    ]


def parse_and_validate_stream(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]], int]:
    """Parses JSONL file into (records_by_key, error_records, schema_failure_count)."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    schema_failures = 0

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                schema_failures += 1
                continue

            if not isinstance(record, dict):
                schema_failures += 1
                continue

            video = record.get("video")
            if not isinstance(video, str) or not video:
                schema_failures += 1
                continue

            if "error" in record:
                if set(record.keys()) != {"video", "error"} or not isinstance(
                    record["error"], str
                ):
                    schema_failures += 1
                    continue
                errors.append({"video": video, "error": record["error"]})
                continue

            if set(record.keys()) != {"video", "second", "objects", "people"}:
                schema_failures += 1
                continue

            second = record["second"]
            people = record["people"]
            objects = record["objects"]

            if not isinstance(second, int) or isinstance(second, bool) or second < 0:
                schema_failures += 1
                continue
            if not isinstance(people, int) or isinstance(people, bool) or people < 0:
                schema_failures += 1
                continue
            if not isinstance(objects, list) or any(
                not isinstance(item, str) or item not in CANONICAL_INDEX for item in objects
            ):
                schema_failures += 1
                continue
            if len(set(objects)) != len(objects):
                schema_failures += 1
                continue
            if objects != sorted(objects, key=CANONICAL_INDEX.__getitem__):
                schema_failures += 1
                continue

            key = (video, second)
            if key in records:
                schema_failures += 1
                continue

            records[key] = record

    return records, errors, schema_failures


def compare_records(
    reference_path: Path,
    candidate_path: Path,
    exact: bool = False,
) -> tuple[dict[str, Any], bool]:
    ref_records, ref_errors, ref_schema_errs = parse_and_validate_stream(reference_path)
    cand_records, cand_errors, cand_schema_errs = parse_and_validate_stream(candidate_path)

    ref_keys = set(ref_records.keys())
    cand_keys = set(cand_records.keys())

    common_keys = ref_keys & cand_keys
    missing_keys = sorted(ref_keys - cand_keys)
    unexpected_keys = sorted(cand_keys - ref_keys)

    total_ref = len(ref_keys)
    coverage = len(common_keys) / total_ref if total_ref > 0 else 1.0

    # Object Micro F1
    tp = 0
    fp = 0
    fn = 0

    for k in ref_keys:
        exp_objs = set(ref_records[k]["objects"])
        if k in cand_records:
            cand_objs = set(cand_records[k]["objects"])
            tp += len(exp_objs & cand_objs)
            fp += len(cand_objs - exp_objs)
            fn += len(exp_objs - cand_objs)
        else:
            fn += len(exp_objs)

    for k in unexpected_keys:
        cand_objs = set(cand_records[k]["objects"])
        fp += len(cand_objs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else (1.0 if (tp == 0 and fp == 0 and fn == 0) else 0.0)
    )

    # People counts
    people_exact = 0
    people_within_one = 0
    exact_matches = 0

    for k in ref_keys:
        if k in cand_records:
            exp_p = ref_records[k]["people"]
            cand_p = cand_records[k]["people"]
            if exp_p == cand_p:
                people_exact += 1
            if abs(exp_p - cand_p) <= 1:
                people_within_one += 1

            # Exact record match: identical objects AND people
            if (
                ref_records[k]["objects"] == cand_records[k]["objects"]
                and exp_p == cand_p
            ):
                exact_matches += 1

    people_exact_rate = people_exact / total_ref if total_ref > 0 else 1.0
    people_within_one_rate = people_within_one / total_ref if total_ref > 0 else 1.0
    exact_record_rate = exact_matches / total_ref if total_ref > 0 else 1.0

    # Error record comparison with normalized (video, error_message) multiset
    ref_error_counts = Counter((e["video"], e["error"]) for e in ref_errors)
    cand_error_counts = Counter((e["video"], e["error"]) for e in cand_errors)
    exact_error_match = cand_error_counts == ref_error_counts

    if exact:
        passed = (
            ref_schema_errs == 0
            and cand_schema_errs == 0
            and len(missing_keys) == 0
            and len(unexpected_keys) == 0
            and exact_record_rate == 1.0
            and exact_error_match
        )
    else:
        # Reference schema failures must fail both exact and threshold modes
        passed = (
            ref_schema_errs == 0
            and cand_schema_errs == 0
            and f1 >= THRESHOLD_OBJECT_F1
            and people_exact_rate >= THRESHOLD_PEOPLE_EXACT
            and people_within_one_rate >= THRESHOLD_PEOPLE_WITHIN_ONE
        )

    metrics = {
        "exact_mode": exact,
        "passed": passed,
        "reference_file": str(reference_path),
        "candidate_file": str(candidate_path),
        "total_expected_records": total_ref,
        "total_candidate_records": len(cand_keys),
        "common_records": len(common_keys),
        "missing_records": len(missing_keys),
        "unexpected_records": len(unexpected_keys),
        "record_coverage": coverage,
        "object_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "micro_f1": f1,
            "threshold": THRESHOLD_OBJECT_F1,
            "passed": (fp == 0 and fn == 0) if exact else (f1 >= THRESHOLD_OBJECT_F1),
        },
        "people_metrics": {
            "exact_matches": people_exact,
            "exact_rate": people_exact_rate,
            "exact_threshold": THRESHOLD_PEOPLE_EXACT,
            "exact_passed": people_exact_rate >= THRESHOLD_PEOPLE_EXACT if not exact else people_exact_rate == 1.0,
            "within_one_matches": people_within_one,
            "within_one_rate": people_within_one_rate,
            "within_one_threshold": THRESHOLD_PEOPLE_WITHIN_ONE,
            "within_one_passed": people_within_one_rate >= THRESHOLD_PEOPLE_WITHIN_ONE if not exact else people_within_one_rate == 1.0,
        },
        "exact_record_match": {
            "matches": exact_matches,
            "rate": exact_record_rate,
        },
        "error_records": {
            "reference_errors": len(ref_errors),
            "candidate_errors": len(cand_errors),
            "exact_error_match": exact_error_match,
            "reference_error_counts": serialize_error_counts(ref_error_counts),
            "candidate_error_counts": serialize_error_counts(cand_error_counts),
        },
        "schema_failures": {
            "reference_failures": ref_schema_errs,
            "candidate_failures": cand_schema_errs,
        },
    }
    return metrics, passed


def print_report(metrics: dict[str, Any]) -> None:
    mode_str = "EXACT MODE" if metrics["exact_mode"] else "DEFAULT THRESHOLD MODE"
    status_str = "PASSED" if metrics["passed"] else "FAILED"
    print("=" * 60)
    print(f"REFERENCE COMPARISON REPORT [{mode_str}] - {status_str}")
    print("=" * 60)
    print(f"Reference:  {metrics['reference_file']}")
    print(f"Candidate:  {metrics['candidate_file']}")
    print("-" * 60)
    print(f"Record Coverage:       {metrics['record_coverage'] * 100:.2f}% ({metrics['common_records']}/{metrics['total_expected_records']})")
    print(f"Missing Records:       {metrics['missing_records']}")
    print(f"Unexpected Records:    {metrics['unexpected_records']}")
    print(f"Exact Record Match:    {metrics['exact_record_match']['rate'] * 100:.2f}%")
    print("-" * 60)
    obj = metrics["object_metrics"]
    print(f"Object Micro F1:       {obj['micro_f1']:.4f} (threshold: {obj['threshold']:.2f}) -> {'PASS' if obj['passed'] else 'FAIL'}")
    print(f"  TP: {obj['true_positives']} | FP: {obj['false_positives']} | FN: {obj['false_negatives']}")
    print("-" * 60)
    peep = metrics["people_metrics"]
    print(f"People Exact Match:    {peep['exact_rate'] * 100:.2f}% (threshold: {peep['exact_threshold'] * 100:.1f}%) -> {'PASS' if peep['exact_passed'] else 'FAIL'}")
    print(f"People Within One:     {peep['within_one_rate'] * 100:.2f}% (threshold: {peep['within_one_threshold'] * 100:.1f}%) -> {'PASS' if peep['within_one_passed'] else 'FAIL'}")
    print("-" * 60)
    err = metrics["error_records"]
    print(f"Candidate Errors:      {err['candidate_errors']} (Reference: {err['reference_errors']}) -> Exact Match: {'YES' if err['exact_error_match'] else 'NO'}")
    print(f"Schema Failures:       Candidate: {metrics['schema_failures']['candidate_failures']}, Reference: {metrics['schema_failures']['reference_failures']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare candidate JSONL results against stable reference answers."
    )
    parser.add_argument("candidate", type=Path, help="Candidate results.jsonl path.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Path to expected reference JSONL file.",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Require exact bit-for-bit record and count equivalence (fail on any difference).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print metrics as JSON.",
    )
    args = parser.parse_args()

    try:
        metrics, passed = compare_records(
            reference_path=args.reference,
            candidate_path=args.candidate,
            exact=args.exact,
        )
        if args.json:
            print(json.dumps(metrics, indent=2))
        else:
            print_report(metrics)
        return 0 if passed else 1
    except Exception as exc:
        print(f"Comparator error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
