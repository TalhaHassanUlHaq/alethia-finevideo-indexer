# Gemini Phase 4 T4 Batch-Size Experiment Result Record

## 1. Overview and Status

- **Experiment**: T4 GPU Batch-Size Progression (`[1, 2, 4, 8, 16, 32]`, with conditional 64)
- **Kernel**: `talhahassanulhaq/althia-batch-size-benchmark` (Version 1)
- **Kaggle Kernel URL**: <https://www.kaggle.com/code/talhahassanulhaq/althia-batch-size-benchmark>
- **Dataset Attached**: `talhahassanulhaq/althia-reference-suite`
- **Kernel Attached**: `talhahassanulhaq/althia-phase-3-reference`
- **Hardware**: NVIDIA Tesla T4 (16 GB VRAM)
- **Terminal Status**: `KernelWorkerStatus.COMPLETE`
- **Execution Order**:
  - Repetition 1: Ascending (`[1, 2, 4, 8, 16, 32]`)
  - Repetition 2: Descending (`[32, 16, 8, 4, 2, 1]`)
  - Repetition 3: Ascending (`[1, 2, 4, 8, 16, 32]`)

---

## 2. Immutable Reference Verification

Reference artifacts were verified before editing and after all operations. Both files remain byte-for-byte identical to their authoritative hashes:

| Artifact | Required SHA-256 | Verified Local SHA-256 | Status |
|---|---|---|---|
| `reference/expected.jsonl` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | Identical |
| `artifacts/reference/input/reference.parquet` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | Identical |
| `yolo11l.pt` (Kaggle reference) | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | Verified |

---

## 3. Files Changed and Deliverables

### Repository Code and Tests
- `src/althia/pipeline.py`: Added explicit validation (`if batch_size < 1: raise ValueError(...)`) in `run_pipeline` in addition to `process_video`.
- `tests/test_pipeline.py`: Added 5 focused unit tests using fakes (no GPU / YOLO dependency) covering:
  - Default CLI batch size is 1.
  - Batch size validation rejects values below 1.
  - Frame timestamps, order, and result pairings are preserved under batching.
  - Batches never cross video boundaries.
  - Partial failure handling remains intact under batching.
- `scripts/benchmark_batch_size.py`: Standalone driver and isolated worker benchmark program with background memory sampling (`torch.cuda.mem_get_info`), exact semantic comparator, and mismatch logging.
- `scripts/build_kaggle_benchmark.py`: Packager script that builds the self-contained Kaggle execution script.
- `kaggle/batch_size/kernel-metadata.json`: Kernel metadata targeting `NvidiaTeslaT4`.
- `kaggle/batch_size/benchmark.py`: Standalone driver script executed on Kaggle.

### Promoted Deliverables
- `benchmarks/batch_size/results.csv`: Complete aggregated metrics across all batch sizes.
- `benchmarks/batch_size/report.md`: Markdown report with executive summary, tables, and findings.
- `benchmarks/batch_size/throughput.png`: Visual chart of throughput, speedup, and GPU memory usage.

### Raw Outputs (under ignored paths)
- `artifacts/kaggle_batch_output/`: Full downloaded Kaggle outputs and job log (`althia-batch-size-benchmark.log`).
- `artifacts/batch_size/`:
  - 18 per-repetition JSONL files (`output_b{1,2,4,8,16,32}_rep{1,2,3}.jsonl`)
  - 18 per-repetition metrics JSON files (`metrics_b{1,2,4,8,16,32}_rep{1,2,3}.json`)
  - 18 per-repetition mismatch JSON files (`mismatch_b{1,2,4,8,16,32}_rep{1,2,3}.json`)
  - 18 worker subprocess logs (`worker_b{1,2,4,8,16,32}_rep{1,2,3}.log`)

---

## 4. Verification Results

### Local Unit Tests
```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```
- **Pytest**: `34 passed in 16.94s` (up from 29 baseline tests, 5 new regression tests).
- **Compileall**: 0 errors across all directories.

### Kaggle In-Kernel Verification
- **Pytest**: `34 passed in 8.36s` before starting the benchmark.

---

## 5. Actual Results Table

| Batch | End-to-end FPS | Speedup | Peak GPU memory | JSONL match | Status |
|---|---:|---:|---:|---|---|
| 1 | 42.45 | 1.00x | 412.9 MiB (97.2% free) | Exact | Completed |
| 2 | 43.55 | 1.03x | 542.9 MiB (96.4% free) | Exact | Completed |
| 4 | 41.55 | 0.98x | 840.9 MiB (94.4% free) | Exact | Completed |
| 8 | 40.01 | 0.94x | 1226.9 MiB (91.8% free) | Exact | Completed |
| 16 | 40.17 | 0.95x | 2184.9 MiB (85.3% free) | Exact | Completed |
| 32 | 38.25 | 0.90x | 4092.9 MiB (72.6% free) | Exact | Completed |

### Detailed Performance Breakdown

| Batch | Runtime (s) | E2E FPS | Ultralytics Inf FPS | Sync Call FPS | Peak Alloc MiB | Peak Reserved MiB | Mixed Shapes | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 176.02 | 42.45 | 51.25 | 45.74 | 222.2 | 256.0 | 0 | Yes |
| 2 | 171.59 | 43.55 | 51.86 | 46.88 | 313.6 | 386.0 | 0 | Yes |
| 4 | 179.85 | 41.55 | 48.85 | 44.58 | 497.5 | 684.0 | 0 | Yes |
| 8 | 186.78 | 40.01 | 46.43 | 42.74 | 860.5 | 1070.0 | 0 | Yes |
| 16 | 186.05 | 40.17 | 46.37 | 42.83 | 1591.3 | 2028.0 | 0 | Yes |
| 32 | 195.35 | 38.25 | 43.82 | 40.62 | 3051.6 | 3936.0 | 0 | Yes |

---

## 6. Batch Size Selections

- **Fastest Measured Batch Size**: **2** (43.55 FPS, 1.03x speedup)
- **Best Strictly Equivalent Batch Size**: **2** (43.55 FPS, 100% exact match across all 3 repetitions)
- **Recommended Production Batch Size**: **1** (42.45 FPS)
  - **Rationale**: Batch 1 achieves 42.45 FPS, which is within **2.5%** of the fastest eligible batch size (Batch 2 at 43.55 FPS). Under the decision criteria ("If another eligible smaller size is within 5% of the fastest eligible FPS, recommend the smaller size"), Batch 1 is recommended.
  - **Key System Insight**: Serial CPU video decoding with PyAV is the dominant bottleneck in this pipeline. Increasing YOLO batch size does not alleviate the decoding bottleneck; instead, larger batches add frame-buffering latency and memory allocation overhead without meaningful GPU throughput scaling.

---

## 7. Observations, Constraints, and Limitations

1. **Exact Correctness**:
   - Every one of the 18 benchmark runs (6 batch sizes × 3 repetitions) achieved **100% exact match** with `reference/expected.jsonl`.
   - Every generated `output_b*_rep*.jsonl` file has identical byte size (22,934 bytes) and identical SHA-256 digest (`7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`).
   - Mismatch files recorded 0 missing keys, 0 unexpected keys, 0 differing records, and micro-F1 = 1.0.

2. **Mixed Shapes and Preprocessing Equivalence**:
   - Exactly **0 mixed-shape inference calls** were observed across all batch sizes.
   - Because batching is bounded strictly within individual video streams, frame dimensions remain homogeneous within each batch, preserving Ultralytics minimum-rectangle letterbox preprocessing.

3. **Memory Headroom & OOM**:
   - Zero OOM errors were encountered.
   - Peak actual device memory usage scaled from 412.9 MiB (batch 1) to 4,092.9 MiB (batch 32).
   - Minimum headroom remained at 72.6% on the 16 GB Tesla T4, well above the 20% safety threshold.

4. **Batch 64 Decision**:
   - Batch 64 was **skipped**.
   - Condition check: Batch 32 median FPS (38.25 FPS) regressed by -4.76% compared to Batch 16 (40.17 FPS), failing the required > 5% improvement criterion.

5. **Decoded Frame Count**:
   - Probing and decoding `reference.parquet` produces exactly **7,473 frames** across 10 videos (316 output seconds). All 7,473 frames were processed without dropped or duplicated frames on every run.

---

## 8. Stopping Condition

All task requirements are fulfilled, deliverables are promoted to `benchmarks/batch_size/`, and git changes remain uncommitted per instructions. Stopping for Codex review.
