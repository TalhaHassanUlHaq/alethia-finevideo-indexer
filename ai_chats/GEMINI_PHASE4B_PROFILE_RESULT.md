# Phase 4B Pipeline Profiling Result

## Overview

- **Task**: Phase 4B T4 Pipeline Timing and Utilization Profile (`ai_chats/GEMINI_TASK_PHASE4B_PIPELINE_PROFILE.md`).
- **Nature of Work**: Empirical measurement and profiling across batch sizes `1`, `2`, `8`, and `32` on an NVIDIA Tesla T4 GPU.
- **Constraints Maintained**: Measurement-only; zero changes to production semantics, zero optimizations introduced (no FP16, no frame skipping, no concurrency, no threshold changes), all reference artifacts preserved, `WRITEUP.md` untouched, no git commits.

---

## 1. Files Created and Modified

- **Profiling Implementation**:
  - `scripts/profile_pipeline.py`: Comprehensive profiling harness supporting isolated worker mode and driver mode. Tracks outer non-overlapping wall timers (Parquet load, PyAV video decode, BGR conversion/rotation, detector wall time, aggregation, JSON writing, and residual) and inner YOLO stages (`preprocess`, `inference`, `postprocess`). Collects NVML GPU utilization and process CPU stats.
  - `tests/test_profile.py`: 4 focused unit tests covering the non-overlapping outer accounting equation, NVML GPU utilization calculation, single-run variability formatting, and decoder frame preservation with fakes.
  - `scripts/build_kaggle_profile.py`: Self-contained builder script bundling workspace code and runner logic into a standalone Kaggle script.
  - `kaggle/pipeline_profile/kernel-metadata.json`: Kernel metadata pinning to Tesla T4 with datasets `talhahassanulhaq/althia-reference-suite` and notebook input `talhahassanulhaq/althia-phase-3-reference`.
  - `kaggle/pipeline_profile/profile.py`: Generated standalone Kaggle runner script.
- **Deliverables**:
  - `benchmarks/pipeline_profile/results.csv`: Complete metrics table across all batch sizes.
  - `benchmarks/pipeline_profile/report.md`: Markdown report with 4 required tables, bottleneck diagnosis, optimization implications, and limitations.
  - `benchmarks/pipeline_profile/timing_breakdown.png`: Stacked bar chart of stage timings.
  - `benchmarks/pipeline_profile/gpu_utilization.png`: Utilization curve and VRAM allocation plot.
  - `artifacts/pipeline_profile/*`: Raw JSON metrics (`profile_metrics_b*.json`), mismatch reports (`profile_mismatch_b*.json`), worker logs (`profile_worker_b*.log`), GPU utilization samples (`gpu_samples_b*.json`), and output JSONLs (`profile_output_b*.jsonl`).

---

## 2. Verification and Integrity Checks

### Reference Artifact Hashes (Before and After)

| Artifact | Authoritative SHA-256 | Measured SHA-256 | Status |
|:---|:---|:---|:---|
| `reference/expected.jsonl` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` | **Exact Match** |
| `artifacts/reference/input/reference.parquet` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` | **Exact Match** |
| `yolo11l.pt` | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` | **Exact Match** |

### Output JSONL Hashes

All 4 profiling runs on Kaggle produced output JSONLs bit-for-bit identical to `reference/expected.jsonl`:

- `profile_output_b1.jsonl`: `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- `profile_output_b2.jsonl`: `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- `profile_output_b8.jsonl`: `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- `profile_output_b32.jsonl`: `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`

Semantic comparison against `reference/expected.jsonl`:
- Completed videos: 10 / 10
- Total inferred frames: 7,473 / 7,473
- JSONL records: 316 / 316
- Differing records: 0
- Error records: 0
- Mixed-shape calls: 0

### Test Suite Execution

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
......................................                                   [100%]
38 passed in 32.41s

python -m compileall -q run.py prepare.py src scripts tests kaggle
# Exited with code 0 (clean compilation)
```

---

## 3. Kaggle Execution Summary

- **Kernel**: `talhahassanulhaq/althia-pipeline-profile`
- **URL**: <https://www.kaggle.com/code/talhahassanulhaq/althia-pipeline-profile>
- **Hardware**: NVIDIA Tesla T4 (15,360 MiB GDDR6), 4 vCPUs, 30 GB RAM
- **Terminal Status**: `KernelWorkerStatus.COMPLETE`
- **Total Execution Time**: ~17.25 minutes (1,035 seconds total container duration)
- **Raw Download Path**: `artifacts/kaggle_profile_output/`

---

## 4. Measured Profile Results

### Table 1: Stage Breakdown Table (Batches 1, 2, 8, 32)

| Batch | End-to-End (s) | Parquet Load (s) | PyAV Decode (s) | Convert/Rotate (s) | YOLO Preprocess (s) | GPU Inference (s) | YOLO Postprocess (s) | Agg + Write (s) | Residual (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 180.74 | 0.036 (0.0%) | 5.05 (2.8%) | 5.83 (3.2%) | 6.75 (3.7%) | 152.23 (84.2%) | 7.31 (4.0%) | 0.21 (0.1%) | 3.33 |
| **2** | 182.43 | 0.035 (0.0%) | 4.93 (2.7%) | 5.96 (3.3%) | 6.73 (3.7%) | 155.98 (85.5%) | 6.58 (3.6%) | 0.16 (0.1%) | 2.05 |
| **8** | 204.05 | 0.038 (0.0%) | 5.03 (2.5%) | 6.18 (3.0%) | 7.31 (3.6%) | 178.23 (87.3%) | 6.16 (3.0%) | 0.09 (0.0%) | 1.02 |
| **32** | 216.14 | 0.037 (0.0%) | 4.69 (2.2%) | 5.92 (2.7%) | 7.49 (3.5%) | 191.72 (88.7%) | 5.63 (2.6%) | 0.05 (0.0%) | 0.61 |

*Note: Combined Decode + Conversion is 10.88s (6.0%) in Batch 1. Residual represents combined detector residual (2.46s) + outer residual (0.87s).*

### Table 2: GPU Utilization and Memory Table

| Batch | Mean SM Util (%) | Median SM Util (%) | p95 SM Util (%) | Max SM Util (%) | Samples >= 90% (%) | Peak VRAM (MiB) | Headroom (%) | Mean Power (W) | Peak Allocated (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 84.7% | 85.0% | 88.0% | 89.0% | 0.0% | 861.2 | 94.4% | 66.1 | 222.2 |
| **2** | 86.3% | 87.0% | 90.0% | 92.0% | 11.1% | 991.2 | 93.5% | 65.4 | 313.6 |
| **8** | 88.5% | 89.0% | 95.9% | 100.0% | 38.0% | 1675.2 | 89.1% | 66.0 | 860.5 |
| **32** | 90.0% | 100.0% | 100.0% | 100.0% | 71.3% | 4541.2 | 70.4% | 63.4 | 3051.6 |

### Table 3: Single-Run Timing and CPU Utilization Table

| Batch | Observed Runtime (s) | E2E FPS | Total CPU Time (s) | CPU/Wall Ratio | Logical CPUs | Runtime Std Dev | Runtime CoV | Between-Run Variability |
|---:|---:|---:|---:|---:|---:|---|---|---|
| **1** | 180.74 | 41.35 | 183.54 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| **2** | 182.43 | 40.96 | 185.75 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| **8** | 204.05 | 36.62 | 207.94 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| **32** | 216.14 | 34.57 | 219.82 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |

*Note: Per task instructions, each batch size was measured in one independent profiling run. Repeat-run timing variability is explicitly marked as `N/A (one run)` rather than fabricating statistics.*

### Table 4: Accounting and Residual Reconciliation Table

**Accounting Equations**:
- Outer wall equation: $T_{\text{wall}} = T_{\text{parquet}} + T_{\text{decode}} + T_{\text{convert}} + T_{\text{detector\_wall}} + T_{\text{agg}} + T_{\text{write}} + T_{\text{outer\_residual}}$
- Detector internal breakdown: $T_{\text{detector\_wall}} = T_{\text{prep}} + T_{\text{inf}} + T_{\text{post}} + T_{\text{detector\_residual}}$

| Batch | Total Wall (s) | Outer Attributed (s) | Outer Residual (s) | Outer Residual (%) | Detector Wall (s) | Detector Attributed (s) | Detector Residual (s) | Detector Residual (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | 180.74 | 179.86 | 0.87 | 0.48% | 168.74 | 166.29 | 2.46 | 1.36% |
| **2** | 182.43 | 181.77 | 0.66 | 0.36% | 170.69 | 169.30 | 1.39 | 0.76% |
| **8** | 204.05 | 203.57 | 0.48 | 0.23% | 192.24 | 191.69 | 0.54 | 0.27% |
| **32** | 216.14 | 215.75 | 0.39 | 0.18% | 205.05 | 204.83 | 0.22 | 0.10% |

Reconciliation is extremely tight across all batch sizes: outer residual is $<0.5\%$ of wall time, and detector internal residual is $<1.4\%$ of detector wall time.

---

## 5. Bottleneck Analysis & Empirical Diagnosis

### Resolving the Open Question: CPU vs GPU Bottleneck

The Phase 4 batch-benchmark report had hypothesized that CPU decoding might be a primary limiter. The empirical profiling data firmly resolves this question:

1. **GPU Inference is the Overwhelming Primary Bottleneck**:
   - In Batch 1, GPU inference consumes **152.23s (84.2% of total wall time)**.
   - Synchronized detector calls represent **168.74s (93.4% of total wall time)**.
   - Meanwhile, PyAV video decoding accounts for only **5.05s (2.8%)** and BGR ndarray conversion/rotation accounts for **5.83s (3.2%)** (total decode + conversion: **10.88s, 6.0%**).
   - Sustained GPU SM utilization is high: **84.7% mean (p95: 88.0%)** on Tesla T4.
   - Therefore, the pipeline is decisively a **GPU-compute bottleneck**, not a CPU decoding bottleneck.

2. **Why Batching Did Not Improve Throughput**:
   - For YOLO11l at 640x640 on Tesla T4, batching increased total inference time from 152.23s (batch 1) to 191.72s (batch 32).
   - T4's SMs are already 85% saturated at batch 1 during FP32 inference.
   - Increasing the batch size increased VRAM pressure (peak allocated went from 222 MiB to 3,051 MiB; peak device memory from 861 MiB to 4,541 MiB) and increased SM saturation (71.3% of samples >= 90% at batch 32) without offering batching parallelism gains in FP32.

3. **Preprocessing, Postprocessing, and IO**:
   - YOLO preprocessing (letterbox, tensor normalization, host-to-device): 6.75s (3.7%).
   - YOLO postprocessing (NMS): 7.31s (4.0%).
   - Aggregation (`add_frame`) and JSON writing: 0.21s (0.1%).
   - Parquet loading: 0.036s (0.0%).
   - These stages represent negligible portions of the runtime.

---

## 6. Evidence-Based Implications for Future Optimizations

1. **FP16 Inference**:
   - **Strongest Expected Lever**: Because GPU inference accounts for 84.2% of total execution time, utilizing FP16 precision directly leverages NVIDIA Tesla T4 Tensor Cores. If FP16 doubles compute throughput without violating accuracy constraints (F1 $\ge 0.95$, people exact count agreement $\ge 85\%$), overall pipeline runtime could drop substantially (the theoretical upper bound on speedup from eliminating half of inference time is $\approx 1.7\times$).
2. **Decoding and Inference Overlap (Asynchronous Pipelining)**:
   - PyAV decoding and BGR conversion consume 10.88s (6.0% of total wall time).
   - In the current single-threaded architecture, CPU decoding and GPU inference are strictly sequential.
   - Overlapping CPU decoding with GPU inference via a background worker thread or queue can hide up to ~10.8s of decode time, yielding a modest ~6% speedup.
3. **Synthesis**:
   - Optimization efforts should prioritize **FP16 inference** as the primary efficiency lever, followed secondarily by **CPU decode overlap** to saturate both compute engines concurrently.

---

## 7. Limitations of the Profile

1. **Reference Fixture Scope**: The 10 videos (315 seconds of video, 7,473 frames) exhibit diverse resolutions, aspect ratios, framerates, and rotation metadata, but may not reflect pathological edge cases in corrupted shards of the larger FineVideo dataset.
2. **Sampling Resolution**: NVML GPU monitoring was sampled every 0.5s (500ms). Brief kernel launches or micro-bursts are smoothed over the 500ms window.
3. **Single Profiling Repetition**: Each batch size was measured once. While execution matched prior Phase 4 benchmark runs within 2.5%, single-run measurements do not quantify run-to-run variance.

---

## 8. Handoff

All deliverables have been created, verified, and placed under their canonical paths. Local tests and compiler checks pass completely. Reference hashes are byte-for-byte identical. No changes have been committed and `WRITEUP.md` has not been altered.

Stopping for Codex review.
