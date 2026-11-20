# Implementation Task: Phase 3 Review Corrections

## Context

Codex reviewed the first Phase 3 implementation. The current 22 tests pass and
the real nine-video fixture was built correctly, but Phase 3 is not yet
accepted. Fix only the confirmed issues below.

## Required corrections

### 1. Stream selected Parquet rows

`scripts/build_reference_fixture.py` currently calls `parquet.read()` for each
selected row, loading an entire approximately 0.5 GB shard into memory. Replace
this with bounded row/batch iteration over only the `mp4` and `json` columns.
Stop reading a shard once all requested rows from it have been collected.
Group selections by source shard so a shard is opened and scanned at most once.
Preserve the exact frozen fixture order and all existing hashes/manifest fields.

Add a test that makes the full-table `ParquetFile.read()` path fail if called,
thereby proving the builder uses bounded iteration rather than a full read.

### 2. Make exact comparison truly exact

In `scripts/compare_reference.py`, `--exact` currently checks only that the
candidate and reference contain the same number of error records. It can pass
when the error records identify different videos or contain different text.

Exact mode must fail on any normalized result or error-record difference,
including different error video IDs, messages, multiplicity, missing errors,
unexpected errors, schema failures, missing records, or unexpected records.
Reference schema failures must fail both exact and threshold modes.

Add regression tests demonstrating that equal error counts with different
content fail exact mode, while identical mixed result/error streams pass.

### 3. Enforce all fixture videos during generation

`scripts/generate_reference.py` calls `validate_jsonl` without
`expected_videos`. Derive the expected video count from all input Parquet
metadata, pass it to validation, and verify the pipeline summary processed that
same number. Reference generation must fail rather than write authoritative
metadata when any fixture video is absent.

Add a unit test using mocks/fakes so YOLO and GPU execution are not required.

## Constraints

- Follow `AGENTS.md`.
- Do not change production code, model settings, candidates, fixture contents,
  existing payload hashes, Phase 2 evidence, `WRITEUP.md`, or `.env`.
- Do not rebuild the real fixture unless required to prove byte-for-byte
  compatibility. If rebuilt, its SHA-256 must remain
  `d4148f19b4633f6066277584954cde9b71436b052a42499d84992e164cef23a3`;
  otherwise stop and report the difference.
- Do not generate reference answers or begin optimization.

## Verification

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-phase3-fixes
python -m compileall -q run.py prepare.py src scripts tests kaggle
python scripts/validate_jsonl.py artifacts/kaggle_output/results-smoke.jsonl --expected-videos 3 --no-errors
```

Update `ai_chats/GEMINI_PHASE3_RESULT.md` with the correction files, new tests,
exact results, and confirmation that production code and the frozen selection
were untouched. Then stop for Codex review.
