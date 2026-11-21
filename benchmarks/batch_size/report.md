# T4 Batch-Size Experiment Benchmark Report

## Executive Summary

- **Fastest Measured Batch Size**: 2
- **Best Strictly Equivalent Batch Size**: 2
- **Recommended Production Batch Size**: 1
- **Batch 64 Evaluated**: Skipped (batch 32 FPS improvement over batch 16 (-4.76%) did not exceed 5%)

## Results Table

| Batch | End-to-end FPS | Speedup | Peak GPU memory | JSONL match | Status |
|---|---:|---:|---:|---|---|
| 1 | 42.45 | 1.00x | 412.9 MiB (97.2% free) | Exact | Completed |
| 2 | 43.55 | 1.03x | 542.9 MiB (96.4% free) | Exact | Completed |
| 4 | 41.55 | 0.98x | 840.9 MiB (94.4% free) | Exact | Completed |
| 8 | 40.01 | 0.94x | 1226.9 MiB (91.8% free) | Exact | Completed |
| 16 | 40.17 | 0.95x | 2184.9 MiB (85.3% free) | Exact | Completed |
| 32 | 38.25 | 0.90x | 4092.9 MiB (72.6% free) | Exact | Completed |

## Detailed Performance Breakdown

| Batch | Runtime (s) | E2E FPS | Ultralytics Inf FPS | Sync Call FPS | Peak Alloc MiB | Peak Reserved MiB | Mixed Shapes | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 176.02 | 42.45 | 51.25 | 45.74 | 222.2 | 256.0 | 0 | Yes |
| 2 | 171.59 | 43.55 | 51.86 | 46.88 | 313.6 | 386.0 | 0 | Yes |
| 4 | 179.85 | 41.55 | 48.85 | 44.58 | 497.5 | 684.0 | 0 | Yes |
| 8 | 186.78 | 40.01 | 46.43 | 42.74 | 860.5 | 1070.0 | 0 | Yes |
| 16 | 186.05 | 40.17 | 46.37 | 42.83 | 1591.3 | 2028.0 | 0 | Yes |
| 32 | 195.35 | 38.25 | 43.82 | 40.62 | 3051.6 | 3936.0 | 0 | Yes |

## Analysis and Findings

1. **Throughput Scaling**: Batch size 1 achieved 42.45 FPS. Increasing batch size to 2 produced a 1.03x speedup.
2. **Strict Correctness**: Ultralytics batching preserved 100% exact semantic equality with `reference/expected.jsonl` across all valid batch sizes with 0 differing records.
3. **Mixed Shapes**: Exactly 0 mixed-shape calls were observed because batching is scoped strictly within individual video boundaries.
4. **Memory Headroom**: All completed batch sizes maintained greater than the required 20% device memory headroom on the 16 GB Tesla T4.
5. **Production Recommendation**: Batch size **1** is selected for production.

## Preprocessing Equivalence

- Pinned Ultralytics minimum-rectangle letterboxing remains strictly equivalent when batching within individual video streams.
- Rotated frames (such as fixture video 9) maintain consistent dimensions across all frames in that stream.
- 0 mixed-shape calls were triggered, avoiding any dynamic fallback to fixed-square letterbox padding.

