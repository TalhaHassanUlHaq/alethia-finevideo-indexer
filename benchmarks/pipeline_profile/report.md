# T4 Pipeline Timing and Utilization Profile Report

## Executive Summary

- **Primary Bottleneck Diagnosis**: GPU-compute bottleneck (GPU inference takes >50% of time and GPU utilization is high).
- **Batch 1 Profile**: End-to-end 180.74s | Decoded Frames: 7,473 (10 videos, 316 JSONL records).
- **GPU Inference Time**: 152.23s (84.2% of total).
- **CPU Decode + Conversion Time**: 10.88s (6.0% of total).
- **Average GPU SM Utilization**: 84.7% (p95: 88.0%).
- **Semantic Correctness**: Exact match (100% agreement, 0 differing records) with `reference/expected.jsonl` across all runs.

## 1. Stage Breakdown Table (Batches 1, 2, 8, 32)

| Batch | End-to-End (s) | Parquet Load (s) | PyAV Decode (s) | Convert/Rotate (s) | YOLO Preprocess (s) | GPU Inference (s) | YOLO Postprocess (s) | Agg + Write (s) | Residual (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 180.74 | 0.036 (0.0%) | 5.05 (2.8%) | 5.83 (3.2%) | 6.75 (3.7%) | 152.23 (84.2%) | 7.31 (4.0%) | 0.21 (0.1%) | 3.33 |
| 2 | 182.43 | 0.035 (0.0%) | 4.93 (2.7%) | 5.96 (3.3%) | 6.73 (3.7%) | 155.98 (85.5%) | 6.58 (3.6%) | 0.16 (0.1%) | 2.05 |
| 8 | 204.05 | 0.038 (0.0%) | 5.03 (2.5%) | 6.18 (3.0%) | 7.31 (3.6%) | 178.23 (87.3%) | 6.16 (3.0%) | 0.09 (0.0%) | 1.02 |
| 32 | 216.14 | 0.037 (0.0%) | 4.69 (2.2%) | 5.92 (2.7%) | 7.49 (3.5%) | 191.72 (88.7%) | 5.63 (2.6%) | 0.05 (0.0%) | 0.61 |

## 2. GPU Utilization and Memory Table

| Batch | Mean SM Util (%) | Median SM Util (%) | p95 SM Util (%) | Max SM Util (%) | Samples >= 90% (%) | Peak VRAM (MiB) | Headroom (%) | Mean Power (W) | Peak Allocated (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 84.7% | 85.0% | 88.0% | 89.0% | 0.0% | 861.2 | 94.4% | 66.1 | 222.2 |
| 2 | 86.3% | 87.0% | 90.0% | 92.0% | 11.1% | 991.2 | 93.5% | 65.4 | 313.6 |
| 8 | 88.5% | 89.0% | 95.9% | 100.0% | 38.0% | 1675.2 | 89.1% | 66.0 | 860.5 |
| 32 | 90.0% | 100.0% | 100.0% | 100.0% | 71.3% | 4541.2 | 70.4% | 63.4 | 3051.6 |

## 3. Single-Run Timing and CPU Utilization Table

| Batch | Observed Runtime (s) | E2E FPS | Total CPU Time (s) | CPU/Wall Ratio | Logical CPUs | Runtime Std Dev | Runtime CoV | Between-Run Variability |
|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | 180.74 | 41.35 | 183.54 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| 2 | 182.43 | 40.96 | 185.75 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| 8 | 204.05 | 36.62 | 207.94 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |
| 32 | 216.14 | 34.57 | 219.82 | 1.02 | 4 | N/A (one run) | N/A (one run) | N/A (one run) |

## 4. Accounting and Residual Reconciliation Table

Outer accounting equation: $T_{\text{wall}} = T_{\text{parquet}} + T_{\text{decode}} + T_{\text{convert}} + T_{\text{detector\_wall}} + T_{\text{agg}} + T_{\text{write}} + T_{\text{outer\_residual}}$

Detector internal breakdown: $T_{\text{detector\_wall}} = T_{\text{prep}} + T_{\text{inf}} + T_{\text{post}} + T_{\text{detector\_residual}}$

| Batch | Total Wall (s) | Outer Attributed (s) | Outer Residual (s) | Outer Residual (%) | Detector Wall (s) | Detector Attributed (s) | Detector Residual (s) | Detector Residual (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 180.74 | 179.86 | 0.87 | 0.48% | 168.74 | 166.29 | 2.46 | 1.36% |
| 2 | 182.43 | 181.77 | 0.66 | 0.36% | 170.69 | 169.30 | 1.39 | 0.76% |
| 8 | 204.05 | 203.57 | 0.48 | 0.23% | 192.24 | 191.69 | 0.54 | 0.27% |
| 32 | 216.14 | 215.75 | 0.39 | 0.18% | 205.05 | 204.83 | 0.22 | 0.10% |

## 5. Bottleneck Analysis & Answers to Core Questions

### Where the Time Actually Goes
The profiling results provide definitive empirical clarity on the pipeline's operational breakdown:
1. **GPU Inference vs CPU Decode**: For Batch 1, GPU inference consumes 152.23s (84.2%), whereas PyAV video decoding and BGR conversion consume 10.88s (6.0%). Synchronized detector calls represent 93.4% of total wall time.
2. **Ultralytics Preprocessing & Postprocessing**: Preprocessing (letterbox, tensor normalization, host-to-device preparation) takes ~6.75s (3.7%), while postprocessing (NMS) takes ~7.31s (4.0%).
3. **IO and Serialization**: Parquet payload loading (0.036s) and JSONL disk output (0.208s) account for less than 0.5% of total time combined.

### Evidence-Based Implications for Optimizations

- **FP16 Inference**: Because GPU inference accounts for a substantial fraction of the compute time, reducing precision to FP16 has high potential to accelerate tensor operations directly on NVIDIA Tensor Cores, provided strict equivalence or assignment thresholds (F1 >= 0.95, people exact >= 85%) are satisfied.
- **Decoding and Inference Overlap (Pipelining)**: Because video decoding (5.05s) and GPU inference (152.23s) are currently executed sequentially within a single thread, overlapping CPU decoding with GPU inference via asynchronous worker queues or background threads can hide most or all of the CPU decode time.

## 6. Limitations of the Profile

- **Ten-Video Reference Fixture**: The 10 videos (315 seconds of video) represent diverse resolutions, framerates, and rotations, but may not reflect edge cases in corrupted shards in the broader dataset.
- **Sampling Resolution**: NVML GPU monitoring was conducted at 0.5s intervals. Extremely brief sub-millisecond kernel bursts or host-to-device transfers are averaged over the 500ms sampling window.
- **Single Run per Size**: Each batch size was measured in one independent profiling run. Repeat-run timing variability is explicitly marked as `N/A (one run)`.

