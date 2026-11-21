# Phase 5 No-Run Release Preparation Result

## Overview

The no-run release-preparation task has been executed in its entirety in strict compliance with `AGENTS.md`, `CODEX.md`, and `ai_chats/GEMINI_CONTEXT.md`.

**Explicit Guarantee**:
**No model inference, GPU workload, benchmark runner, Kaggle kernel, Kaggle API call, or FineVideo dataset download was initiated during this task.** All modifications were strictly confined to code corrections, validation hardening, reporting truthfulness, and operational handoff documentation.

---

## 1. Reference Artifact Invariants

Authoritative reference hashes remain 100% byte-for-byte unchanged:

| Artifact | Authoritative SHA-256 | Current Hash | Status |
|:---|:---|:---|:---:|
| `reference/expected.jsonl` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | **MATCH** |
| `artifacts/reference/input/reference.parquet` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | **MATCH** |
| `artifacts/kaggle_output/althia/yolo11l.pt` | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | **MATCH** |

---

## 2. Unit Testing and Compilation Verification

All verification checks were run locally using CPU-safe commands and workspace-local temporary directories:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

- **Pytest Result**: **53 passed in 7.32s** (100% pass rate across the full test suite).
  - All existing unit, pipeline, and decoder tests passed.
  - All 9 prediction comparator tests passed (covering exact matches, FP16-style diffs, malformed JSON, invalid keys, duplicate keys, unknown classes, non-canonical ordering, and error-record discrepancies).
- **Compileall Result**: Clean compilation with 0 syntax or import errors across all modules.

---

## 3. Files Changed and Created

### Code and Pipeline Correctness
- `scripts/compare_predictions.py`:
  - Added line-by-line JSON parse error handling (`cand_parse_errors`).
  - Added strict record schema validation (enforcing exact allowed keys for results and errors).
  - Added duplicate `(video, second)` key tracking and reporting.
  - Added unknown class detection and canonical object order validation.
  - Added bidirectional error record matching (`ref_err_map == cand_err_map`).
  - Guaranteed that any parse error, schema violation, duplicate key, order violation, unknown class, or error mismatch strictly forces `exact_match=False`.
- `tests/test_compare_predictions.py`:
  - Added 7 new focused test cases validating failure of `exact_match` under malformed JSON, missing keys, duplicate keys, unknown classes, reversed object order, and differing error messages, as well as verification that identical error records pass exact match.
- `scripts/benchmark_configurations_driver.py`:
  - Rewrote `generate_report(...)` to dynamically derive all numbers, rankings, tables, and answers directly from result rows.
  - Eliminated static/contradictory claims and stale decoder FPS numbers.
  - Separated findings into clearly labeled scopes: **Screening Scope** (one 10-video pass) and **Finalist Scope** (median of 3 independent 10-video repetitions).
  - Explicitly marked where 3-repeat finalist data is unavailable for a candidate (e.g. D1 or D3) as a screening result rather than a finalist conclusion.
  - Explicitly stated that actual assignment accuracy on true ground truth remains unverified.
  - Corrected finalist speedup calculation in `run_driver` to `baseline_finalist_median_runtime / candidate_finalist_median_runtime`, ensuring the D0 baseline finalist is strictly `1.000x`.

### Documentation and Deliverables
- `benchmarks/optimization/summary.csv`:
  - Updated finalist speedup values to match the correct runtime ratio methodology:
    - `cfg_01_D0_fp32_b1`: 172.92s median -> **1.000x**
    - `cfg_26_D2_fp32_b2`: 162.49s median -> **1.064x**
    - `cfg_33_D2_fp16_b4`: 98.47s median -> **1.756x**
- `benchmarks/optimization/report.md`:
  - Regenerated dynamically from actual `raw_results.csv` and corrected `summary.csv` rows.
- `RUN_ON_FRIEND.md`:
  - Authored a factual, end-to-end operational guide for the GPU owner to execute the full 50-shard run.
  - Specified the exact production command:
    ```bash
    python run.py --input <folder-with-50-parquet-shards> --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 16
    ```
  - Included environment setup, `prepare.py`, shard count check, GPU diagnostics, timestamped run logging, output validation, `results.jsonl.gz` compression, `HARDWARE.md` telemetry capture, and safe return artifact rules.
  - Explicitly noted that the candidate must author `WRITEUP.md` themselves without AI assistance.
- `README.md`:
  - Added a reference to `RUN_ON_FRIEND.md` for full-dataset GPU execution while preserving the default baseline command as the compatibility entry point.

---

## 4. Remaining Actions for the Friend / GPU Host

When the repository is transferred to the GPU host:

1. **Environment Setup**:
   Create a Python 3.11 virtual environment and install dependencies (`pip install -r requirements.txt`).
2. **Model Weight Preparation**:
   Run `python prepare.py` with internet access to download and verify `yolo11l.pt`.
3. **Data Verification**:
   Verify all 50 FineVideo shards (`shard_000.parquet` to `shard_049.parquet`) are present.
4. **Execution**:
   Run the production command with logging:
   ```bash
   python run.py --input /path/to/finevideo/data --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 16 2>&1 | tee run.log
   ```
5. **Validation and Packaging**:
   - Run `python scripts/validate_jsonl.py results.jsonl`.
   - Compress to `gzip -k results.jsonl` (creating `results.jsonl.gz`).
   - Compute SHA-256 hashes (`sha256sum results.jsonl results.jsonl.gz`).
6. **Telemetry & Artifact Transfer**:
   - Record GPU, CPU, RAM, and wall-clock runtime for `HARDWARE.md`.
   - Transfer back `results.jsonl.gz`, `run.log`, checksums, and hardware facts.
7. **Writeup**:
   - The candidate authors `WRITEUP.md` independently based on empirical results.

---

## 5. Status

All requirements of `ai_chats/GEMINI_TASK_PHASE5_NO_RUN_RELEASE_PREP.md` are complete. No Git commits have been made. Stopped for Codex review.
