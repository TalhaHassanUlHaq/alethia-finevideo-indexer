# Implementation Task: Phase 3 Final Review Corrections

## Context

Codex independently ran the current suite: 25 tests pass, compilation succeeds,
and the real fixture still hashes to
`d4148f19b4633f6066277584954cde9b71436b052a42499d84992e164cef23a3`.
Three remaining implementation defects must be fixed before GPU generation.

## Required corrections

### 1. Actually bound Parquet video memory

The builder now uses `iter_batches`, but `batch_size=64` loads all 33 large
video rows in each FineVideo shard at once. Change extraction to request one row
per batch while scanning each shard at most once and stopping after its last
selected row. Continue requesting only `mp4` and `json`.

Strengthen the regression test to assert `iter_batches` is called with
`batch_size=1` and exactly those two columns. Retain the existing prohibition on
`ParquetFile.read()`.

### 2. Make comparator reports JSON-safe and internally consistent

`compare_records()` currently puts `Counter` dictionaries with tuple keys into
the metrics object. `scripts/compare_reference.py --json` therefore fails when
error records exist. Represent normalized error multiplicities using a
JSON-serializable deterministic structure while preserving exact multiset
comparison.

Also correct exact-mode `object_metrics.passed`: an exact stream containing no
positive objects must report object comparison as passed when FP and FN are
both zero. Add tests that serialize the full metrics object for identical mixed
result/error streams and for an exactly equal all-empty-object stream.

### 3. Preserve authoritative references on generation failure

`generate_reference()` currently sends `run_pipeline` directly to
`reference/expected.jsonl`. A failed or incomplete generation can overwrite a
previous stable reference before validation fails.

Generate JSONL and metadata through temporary files in the destination
directories. Validate the temporary JSONL, verify all expected videos and zero
errors, and build metadata before atomically replacing the authoritative
outputs. On any failure, remove only the temporary files created by that run
and leave any existing `expected.jsonl` and `metadata.json` byte-for-byte
unchanged.

Add a unit test with sentinel authoritative files proving an incomplete run
raises and preserves both sentinels. Retain the existing successful-generation
test.

## Constraints

- Follow `AGENTS.md`.
- Do not change production code, frozen candidates, model settings, Phase 2
  evidence, `WRITEUP.md`, or `.env`.
- Do not generate reference answers or begin optimization.
- Rebuild the real fixture with `--force` after the batch-size correction and
  verify its SHA-256 remains exactly
  `d4148f19b4633f6066277584954cde9b71436b052a42499d84992e164cef23a3`.

## Verification

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-phase3-final
python -m compileall -q run.py prepare.py src scripts tests kaggle
python scripts/validate_jsonl.py artifacts/kaggle_output/results-smoke.jsonl --expected-videos 3 --no-errors
```

Update `ai_chats/GEMINI_PHASE3_RESULT.md` with the final corrections, new tests,
exact results, fixture hash, and confirmation that production code and frozen
candidates were untouched. Then stop for Codex review.
