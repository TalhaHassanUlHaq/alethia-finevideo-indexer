# Final Submission CLI Defaults Result

## Overview

The final submission CLI defaults task (`ai_chats/GEMINI_TASK_FINAL_SUBMISSION_DEFAULTS.md`) has been fully implemented and verified using CPU-safe tests and compilation.

The required unflagged command:

```bash
python run.py --input <folder> --output results.jsonl
```

now defaults directly to the validated full-run production configuration:
- Decoder strategy: `D2` (bounded prefetch)
- Inference precision: `fp16`
- Batch size: `8`
- Prefetch capacity: `64`

The explicit correctness-first baseline remains fully available via:

```bash
python run.py --input <folder> --output results.jsonl --decoder-strategy D0 --precision fp32 --batch-size 1
```

---

## Explicit Safety & Boundary Confirmations

- **No inference**: No PyTorch or YOLO inference was run.
- **No model weight loading**: `yolo11l.pt` was not loaded or modified.
- **No Kaggle operations**: No Kaggle kernels, APIs, or sessions were invoked.
- **No credentials / secrets**: `.env` and API tokens were not accessed, copied, or printed.
- **Untouched deliverables**: `WRITEUP.md`, `results.jsonl.gz`, `results.jsonl`, `results_hashes.txt`, and reference fixtures were completely untouched.
- **No Git commits**: No commits or git state alterations were made.

---

## Files Modified

1. **`src/althia/config.py`**:
   - Updated dataclass field defaults in `PipelineConfig`:
     - `decoder_strategy: DecoderStrategy = "D2"` (was `"D0"`)
     - `precision: Precision = "fp16"` (was `"fp32"`)
     - `batch_size: int = 8` (was `1`)
     - Retained `prefetch_capacity: int = 64` and `decode_threads: int = 2`.

2. **`src/althia/cli.py`**:
   - Updated CLI argument parser defaults in `parse_args()`:
     - `--decoder-strategy`: default `"D2"`
     - `--precision`: default `"fp16"`
     - `--batch-size`: default `8`
     - Retained `--prefetch-capacity` default `64`.

3. **`tests/test_config.py`**:
   - Updated `test_pipeline_config_defaults()` to assert defaults `D2`, `fp16`, batch size `8`, prefetch capacity `64`.
   - Added `test_pipeline_config_explicit_baseline()` to verify that specifying `D0`, `fp32`, batch size `1` preserves the exact baseline configuration.

4. **`tests/test_pipeline.py`**:
   - Replaced `test_default_cli_batch_size_is_one` with:
     - `test_default_cli_options_use_optimized_defaults()`: verifies default CLI options are `D2`, `fp16`, batch `8`, prefetch `64`.
     - `test_cli_explicit_baseline_flags()`: verifies passing `--decoder-strategy D0 --precision fp32 --batch-size 1` properly sets baseline options.

5. **`README.md`**:
   - Updated `## Run` documentation to accurately describe the default command as the optimized production configuration (`D2`, `fp16`, batch size `8`).
   - Added explicit documentation for running the baseline via `--decoder-strategy D0 --precision fp32 --batch-size 1`.

---

## Verification Results

Executed CPU-safe test suite and compilation in accordance with task instructions:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-final-defaults
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

- **Pytest Output**:
  ```
  .....................................................................    [100%]
  69 passed in 32.99s
  ```
  All 69 unit, config, CLI, decoder, aggregation, and distributed tests passed cleanly.
- **Compileall Output**:
  ```
  Clean (exit code 0 across run.py prepare.py src scripts tests kaggle)
  ```

Stopping for Codex review.
