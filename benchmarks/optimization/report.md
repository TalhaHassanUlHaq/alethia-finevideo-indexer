# T4 FineVideo Optimization Experiment Report

## Executive Summary

### Screening Scope (10-Video Single Controlled Pass)
- **Compatibility Baseline (cfg_01_D0_fp32_b1)**: 42.19 FPS (100% exact match to reference).
- **Fastest Strictly Equivalent**: cfg_26_D2_fp32_b2 (D2 fp32 b2) at 45.52 FPS (1.08x speedup vs screening baseline, 100% exact match).
- **Fastest FP16 / Overall**: cfg_33_D2_fp16_b4 (D2 fp16 b4) at 72.65 FPS (1.72x speedup vs screening baseline | Micro F1 vs ref: 1.0000, People exact: 99.68%, Within-1: 100.0%).

### Finalist Scope (3 Independent 10-Video Repetitions, Isolated Environment)
- **Baseline Finalist (cfg_01_D0_fp32_b1)**: 43.22 FPS median (172.92s median runtime, 1.000x baseline, CoV 1.24%).
- **Fastest Strictly Equivalent Finalist**: cfg_26_D2_fp32_b2 (D2 fp32 b2) at 45.99 FPS median (162.49s median runtime, 1.064x speedup vs finalist baseline, CoV 0.51%, 100% exact match).
- **Fastest Overall Finalist**: cfg_33_D2_fp16_b4 (D2 fp16 b4) at 75.89 FPS median (98.47s median runtime, 1.756x speedup vs finalist baseline, CoV 2.12% | Micro F1: 1.0000, People exact: 99.68%, Within-1: 100.0%).

## Core Findings & Answers to Required Questions

### 1. Fastest Decoding Strategy
In standalone decoding benchmarks, bounded prefetch decoders achieved the highest throughput: D3 (threaded PyAV AUTO threads=2 + prefetch capacity=64) reached 470.98 FPS (1.470x vs D0) and D2 (serial PyAV + prefetch capacity=64) reached 462.57 FPS (1.444x vs D0). In end-to-end pipeline screening, bounded prefetch decoders (D2 and D3) achieved the highest overall throughput across both FP32 and FP16 by completely decoupling CPU frame extraction from GPU inference.

### 2. Effect of PyAV Threading on Decoding
In standalone decoding, setting `codec_context.thread_type = 'AUTO'` with `thread_count = 2` improved PyAV decoding throughput from 320.40 FPS (D0) to 355.08 FPS (1.108x speedup) with 100% pixel-level exact equivalence. Increasing to 4 threads degraded throughput to 344.64 FPS due to vCPU thread contention. Setting `thread_type = 'FRAME'` degraded performance below baseline (307.45 FPS for 2 threads, 297.74 FPS for 4 threads) due to synchronization overhead on variable-resolution FineVideo clips.

In end-to-end pipeline screening without prefetch, D1 multithreading alone yielded modest gains over D0 (e.g. 42.72 vs 42.19 FPS at FP32 b1; 68.05 vs 67.24 FPS at FP16 b4).

### 3. Effect of Prefetch on End-to-End FPS
Prefetching (D2/D3) maintains a bounded queue of decoded frames in memory, decoupling CPU decode from GPU inference:
- In FP32 (GPU-bound, ~85% GPU inference time): Prefetch provides a modest gain (e.g. D0 fp32 b2 at 43.35 FPS vs D2 fp32 b2 at 45.52 FPS in screening, a 5.0% gain; 45.99 FPS median across 3 finalist repetitions) by hiding CPU decode time during inference.
- In FP16 (where GPU inference is accelerated): Prefetch delivers an 8.0% gain over unbuffered decoding (D0 fp16 b4 at 67.24 FPS vs D2 fp16 b4 at 72.65 FPS in screening; 75.89 FPS median across 3 finalist repetitions), preventing GPU starvation.

### 4. FP16 Inference and End-to-End Change
FP16 utilizes the NVIDIA Tesla T4's dedicated Turing Tensor Cores:
- Inference throughput: At batch size 4, GPU inference throughput increased from 46.61 FPS in FP32 (`cfg_03_D0_fp32_b4`) to 73.24 FPS in FP16 (`cfg_33_D2_fp16_b4`), a 1.57x increase in raw inference rate.
- End-to-end throughput: Screening throughput increased from 42.19 FPS (`cfg_01_D0_fp32_b1`) to 72.65 FPS (`cfg_33_D2_fp16_b4`), a 1.72x speedup. In finalist repetitions, `cfg_33` achieved 75.89 FPS median, representing a 1.756x runtime speedup over the 43.22 FPS finalist baseline.

### 5. Best FP16 Batch Size on T4
In screening, batch size 4 achieved the highest FP16 throughput across all decoders (D2 b4 reached 72.65 FPS, D3 b4 reached 72.37 FPS, D1 b4 reached 68.05 FPS, D0 b4 reached 67.24 FPS). Batch size 8 followed at 69.12 FPS for D2/D3. Batch sizes 16 and 32 dropped to ~65-66 FPS. Batch size 4 saturates the T4 Tensor Cores without incurring higher batch-assembly overhead or memory pressure.

### 6. Strictly Equivalent Configurations
All 24 FP32 configurations (D0, D1, D2, D3 across batches 1, 2, 4, 8, 16, 32) achieved 100% exact semantic equality with `reference/expected.jsonl` (0 differing records, 0 mismatches out of 316). None of the FP16 configurations produced byte-for-byte exact matches with the FP32 reference.

### 7. Configurations with Actual Assignment Accuracy Verified
None. The FineVideo reference fixture contains model predictions from the FP32 baseline, not human-annotated ground truth. While FP16 configurations achieve high agreement against this reference (Micro F1: 1.0000, People exact: 99.68%, People within-one: 100.0%), actual task accuracy on true ground truth remains strictly unverified without official evaluator labels.

### 8. Best Measured T4 Configuration
- **Strict Equivalence Recommendation (Zero Accuracy Risk)**: `cfg_26_D2_fp32_b2` (45.52 FPS screening; 45.99 FPS finalist median, 1.064x speedup, 100% exact match, CoV 0.51%).
- **Maximum Throughput Recommendation (Under Assignment Thresholds)**: `cfg_33_D2_fp16_b4` (72.65 FPS screening; 75.89 FPS finalist median, 1.756x speedup, 99.68% people exact, 100% within-one, CoV 2.12%).
- *Note on D1*: The fastest screening D1 configuration was `cfg_21_D1_fp16_b4` (68.05 FPS screening). Because 3-repeat finalist repetitions have not yet been run for D1 on Kaggle, its performance is documented as a screening result only.

### 9. Configurations to Validate First on RTX 4090 (Unmeasured on 4090)
RTX 4090 performance remains completely unmeasured in this experiment. On an NVIDIA RTX 4090 (24 GB VRAM, Ada Lovelace architecture, 4th-gen Tensor Cores, dual NVDEC), validate `D3 FP16 Batch 16 and Batch 32` first. The RTX 4090's vastly greater memory bandwidth (1,008 GB/s vs T4's 320 GB/s) and compute capacity are expected to push the optimal batch size higher than the T4's batch 4.

### 10. Limitations of the Experiment
- Tested on the 10-video reference fixture (7,473 frames); does not evaluate corrupted or damaged video shards from the full 50-shard dataset.
- NVML telemetry sampled at 500ms intervals.
- Screening was conducted in 1 controlled pass per configuration; 3 repetitions were performed for measured finalists.
- CPU-only decoders were evaluated (no hardware NVDEC decoding).

## Summary of Finalist Results (3 Repetitions)

| Config ID | Decoder | Precision | Batch | Median Runtime (s) | Runtime Std (s) | Runtime CoV (%) | Median FPS | Speedup vs Baseline | Exact Match | Micro F1 | People Exact (%) | People Within-1 (%) | Peak VRAM (MiB) |
|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| cfg_01_D0_fp32_b1 | D0 | fp32 | 1 | 172.92 | 2.143 | 1.24% | 43.22 | 1.000x | Yes | 1.0000 | 100.00% | 100.00% | 887.2 |
| cfg_26_D2_fp32_b2 | D2 | fp32 | 2 | 162.49 | 0.834 | 0.51% | 45.99 | 1.064x | Yes | 1.0000 | 100.00% | 100.00% | 991.2 |
| cfg_33_D2_fp16_b4 | D2 | fp16 | 4 | 98.47 | 2.087 | 2.12% | 75.89 | 1.756x | No | 1.0000 | 99.68% | 100.00% | 1019.2 |

## Summary of Screening Results (48 Configurations)

| Config ID | Decoder | Precision | Batch | E2E FPS | Inference FPS | Peak VRAM (MiB) | Exact Match | Micro F1 | Status |
|:---|:---:|:---:|---:|---:|---:|---:|:---:|---:|:---:|
| cfg_07_D0_fp16_b1 | D0 | fp16 | 1 | 40.04 | 43.45 | 787.2 | No | 1.0000 | SUCCESS |
| cfg_19_D1_fp16_b1 | D1 | fp16 | 1 | 39.83 | 42.68 | 787.2 | No | 1.0000 | SUCCESS |
| cfg_31_D2_fp16_b1 | D2 | fp16 | 1 | 41.68 | 42.01 | 787.2 | No | 1.0000 | SUCCESS |
| cfg_43_D3_fp16_b1 | D3 | fp16 | 1 | 41.62 | 41.96 | 787.2 | No | 1.0000 | SUCCESS |
| cfg_01_D0_fp32_b1 | D0 | fp32 | 1 | 42.19 | 46.03 | 887.2 | Yes | 1.0000 | SUCCESS |
| cfg_13_D1_fp32_b1 | D1 | fp32 | 1 | 42.72 | 45.99 | 887.2 | Yes | 1.0000 | SUCCESS |
| cfg_25_D2_fp32_b1 | D2 | fp32 | 1 | 44.04 | 44.43 | 887.2 | Yes | 1.0000 | SUCCESS |
| cfg_37_D3_fp32_b1 | D3 | fp32 | 1 | 44.17 | 44.54 | 887.2 | Yes | 1.0000 | SUCCESS |
| cfg_08_D0_fp16_b2 | D0 | fp16 | 2 | 62.49 | 70.66 | 877.2 | No | 1.0000 | SUCCESS |
| cfg_20_D1_fp16_b2 | D1 | fp16 | 2 | 63.70 | 70.91 | 877.2 | No | 1.0000 | SUCCESS |
| cfg_32_D2_fp16_b2 | D2 | fp16 | 2 | 67.92 | 68.56 | 877.2 | No | 1.0000 | SUCCESS |
| cfg_44_D3_fp16_b2 | D3 | fp16 | 2 | 66.61 | 67.24 | 877.2 | No | 1.0000 | SUCCESS |
| cfg_02_D0_fp32_b2 | D0 | fp32 | 2 | 43.35 | 47.26 | 991.2 | Yes | 1.0000 | SUCCESS |
| cfg_14_D1_fp32_b2 | D1 | fp32 | 2 | 43.73 | 47.11 | 991.2 | Yes | 1.0000 | SUCCESS |
| cfg_26_D2_fp32_b2 | D2 | fp32 | 2 | 45.52 | 45.82 | 991.2 | Yes | 1.0000 | SUCCESS |
| cfg_38_D3_fp32_b2 | D3 | fp32 | 2 | 45.44 | 45.74 | 991.2 | Yes | 1.0000 | SUCCESS |
| cfg_09_D0_fp16_b4 | D0 | fp16 | 4 | 67.24 | 76.38 | 1019.2 | No | 1.0000 | SUCCESS |
| cfg_21_D1_fp16_b4 | D1 | fp16 | 4 | 68.05 | 76.08 | 1019.2 | No | 1.0000 | SUCCESS |
| cfg_33_D2_fp16_b4 | D2 | fp16 | 4 | 72.65 | 73.24 | 1019.2 | No | 1.0000 | SUCCESS |
| cfg_45_D3_fp16_b4 | D3 | fp16 | 4 | 72.37 | 72.95 | 1019.2 | No | 1.0000 | SUCCESS |
| cfg_03_D0_fp32_b4 | D0 | fp32 | 4 | 42.99 | 46.61 | 1313.2 | Yes | 1.0000 | SUCCESS |
| cfg_15_D1_fp32_b4 | D1 | fp32 | 4 | 43.28 | 46.45 | 1313.2 | Yes | 1.0000 | SUCCESS |
| cfg_27_D2_fp32_b4 | D2 | fp32 | 4 | 44.97 | 45.20 | 1313.2 | Yes | 1.0000 | SUCCESS |
| cfg_39_D3_fp32_b4 | D3 | fp32 | 4 | 44.83 | 45.06 | 1313.2 | Yes | 1.0000 | SUCCESS |
| cfg_10_D0_fp16_b8 | D0 | fp16 | 8 | 63.76 | 71.58 | 1229.2 | No | 1.0000 | SUCCESS |
| cfg_22_D1_fp16_b8 | D1 | fp16 | 8 | 64.60 | 71.36 | 1229.2 | No | 1.0000 | SUCCESS |
| cfg_34_D2_fp16_b8 | D2 | fp16 | 8 | 69.12 | 69.59 | 1229.2 | No | 1.0000 | SUCCESS |
| cfg_46_D3_fp16_b8 | D3 | fp16 | 8 | 69.12 | 69.58 | 1229.2 | No | 1.0000 | SUCCESS |
| cfg_04_D0_fp32_b8 | D0 | fp32 | 8 | 41.55 | 44.72 | 1613.2 | Yes | 1.0000 | SUCCESS |
| cfg_16_D1_fp32_b8 | D1 | fp32 | 8 | 41.76 | 44.52 | 1613.2 | Yes | 1.0000 | SUCCESS |
| cfg_28_D2_fp32_b8 | D2 | fp32 | 8 | 43.71 | 43.90 | 1613.2 | Yes | 1.0000 | SUCCESS |
| cfg_40_D3_fp32_b8 | D3 | fp32 | 8 | 43.61 | 43.80 | 1613.2 | Yes | 1.0000 | SUCCESS |
| cfg_11_D0_fp16_b16 | D0 | fp16 | 16 | 59.45 | 66.02 | 1737.2 | No | 1.0000 | SUCCESS |
| cfg_23_D1_fp16_b16 | D1 | fp16 | 16 | 60.54 | 66.29 | 1737.2 | No | 1.0000 | SUCCESS |
| cfg_35_D2_fp16_b16 | D2 | fp16 | 16 | 65.25 | 65.69 | 1737.2 | No | 1.0000 | SUCCESS |
| cfg_47_D3_fp16_b16 | D3 | fp16 | 16 | 64.96 | 65.37 | 1737.2 | No | 1.0000 | SUCCESS |
| cfg_05_D0_fp32_b16 | D0 | fp32 | 16 | 42.12 | 45.27 | 2363.2 | Yes | 1.0000 | SUCCESS |
| cfg_17_D1_fp32_b16 | D1 | fp32 | 16 | 42.42 | 45.18 | 2363.2 | Yes | 1.0000 | SUCCESS |
| cfg_29_D2_fp32_b16 | D2 | fp32 | 16 | 44.30 | 44.51 | 2363.2 | Yes | 1.0000 | SUCCESS |
| cfg_41_D3_fp32_b16 | D3 | fp32 | 16 | 44.29 | 44.48 | 2363.2 | Yes | 1.0000 | SUCCESS |
| cfg_12_D0_fp16_b32 | D0 | fp16 | 32 | 59.83 | 66.36 | 2601.2 | No | 1.0000 | SUCCESS |
| cfg_24_D1_fp16_b32 | D1 | fp16 | 32 | 61.16 | 66.94 | 2601.2 | No | 1.0000 | SUCCESS |
| cfg_36_D2_fp16_b32 | D2 | fp16 | 32 | 66.64 | 67.20 | 2601.2 | No | 1.0000 | SUCCESS |
| cfg_48_D3_fp16_b32 | D3 | fp16 | 32 | 66.36 | 66.88 | 2601.2 | No | 1.0000 | SUCCESS |
| cfg_06_D0_fp32_b32 | D0 | fp32 | 32 | 40.75 | 43.66 | 3675.2 | Yes | 1.0000 | SUCCESS |
| cfg_18_D1_fp32_b32 | D1 | fp32 | 32 | 41.07 | 43.58 | 3675.2 | Yes | 1.0000 | SUCCESS |
| cfg_30_D2_fp32_b32 | D2 | fp32 | 32 | 43.20 | 43.44 | 3675.2 | Yes | 1.0000 | SUCCESS |
| cfg_42_D3_fp32_b32 | D3 | fp32 | 32 | 43.00 | 43.21 | 3675.2 | Yes | 1.0000 | SUCCESS |
