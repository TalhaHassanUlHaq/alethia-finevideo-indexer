# Gemini Phase 2 Result Record

## Files changed

- `CURRENT_PIPELINE.md`: Corrected the JSON output example in section 5 to use the key `"video"` instead of `"video_id"`.
- `tests/test_aggregation.py`: Added tests verifying that missing elapsed seconds are emitted with `objects: []` and `people: 0`, and that the second containing the final decoded frame is emitted even when it is a partial second.
- `tests/test_pipeline.py`: Added an end-to-end test with temporary Parquet files and mock decoders/detectors testing `<Parquet basename>:<zero-based row>` video ID formatting, partway failure retaining prior second records and emitting a short error record, fault-tolerant continuation to subsequent videos, and validation of the produced mixed JSONL file.
- `tests/test_validation.py`: Added a test verifying `validate_jsonl` on mixed valid and error records when `allow_errors=True` and `allow_errors=False`.
- `pytest.ini`: Added pytest configuration (`testpaths = tests`, `norecursedirs = artifacts .*`, `cache_dir = artifacts/pytest_cache`) to prevent pytest from scanning locked/permission-restricted directory trees under `artifacts/` during discovery.

## Tests added

1. `tests/test_aggregation.py::test_missing_elapsed_second_emitted_with_empty_objects_and_zero_people`
   - Proves that an elapsed second without any decoded frames is emitted with `objects: []` and `people: 0`.
2. `tests/test_aggregation.py::test_partial_second_holding_final_frame_is_emitted`
   - Proves that the second containing the final frame (e.g. at elapsed time 1.05s) is emitted even when it is a partial second.
3. `tests/test_pipeline.py::test_pipeline_video_ids_fault_tolerance_and_validation`
   - Proves video identifiers are exactly `<Parquet basename>:<zero-based row>` across multiple parquet shards and rows (`shard_000.parquet:0`, `shard_000.parquet:1`, `shard_001.parquet:0`).
   - Proves a video failing partway retains its completed seconds (`second: 0`) and emits a single short error record (`{"video": "shard_000.parquet:1", "error": "RuntimeError: stream decode failed partway"}`).
   - Proves pipeline processing continues to the next video (`shard_001.parquet:0`) after a damaged video.
   - Proves the resulting mixed result/error JSONL passes `validate_jsonl` with `allow_errors=True` (5 lines, 3 videos, 1 error) and raises `ValueError` with `allow_errors=False`.
4. `tests/test_validation.py::test_validate_jsonl_mixed_results_and_errors`
   - Directly proves `validate_jsonl` validates contiguous seconds per video in a mixed JSONL file with error records when `allow_errors=True` and raises `ValueError` when `allow_errors=False`.

## Whether production code changed and why

No production code in `src/`, `run.py`, or `prepare.py` was changed. The existing implementation already conforms to all required behaviors, and no defects were exposed by the new tests.

## Exact verification results

1. Pytest suite:
   ```powershell
   python -m pytest -q --basetemp artifacts/pytest-gemini-phase2
   ```
   Output:
   ```
   .............                                                            [100%]
   13 passed in 1.14s
   ```
   Exit code: `0`

2. Compilation check:
   ```powershell
   python -m compileall -q run.py prepare.py src scripts tests kaggle
   ```
   Output: None (compiled successfully)
   Exit code: `0`

3. Kaggle smoke result JSONL validation:
   ```powershell
   python scripts/validate_jsonl.py artifacts/kaggle_output/results-smoke.jsonl --expected-videos 3 --no-errors
   ```
   Output:
   ```json
   {"errors": 0, "lines": 105, "videos": 3}
   ```
   Exit code: `0`
