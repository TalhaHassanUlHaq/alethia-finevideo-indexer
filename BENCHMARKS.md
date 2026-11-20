# Benchmark Record

This file records measured engineering data. It is not a draft of
`WRITEUP.md`; the candidate remains the sole author of the write-up answers.

## Current all-frame FP32 baseline

Verified Phase 3 reference run:

| Measurement | Value |
|---|---:|
| GPU | NVIDIA Tesla T4 |
| Reference footage | 315 metadata seconds |
| Output records | 316 |
| Videos | 10 |
| Errors | 0 |
| Wall time | 170.656215052 seconds |
| Batch size | 1 |
| Precision | FP32 |

Full local FineVideo input:

| Measurement | Value |
|---|---:|
| Parquet shards | 50 |
| Videos | 1,650 |
| Input bytes | 24,822,988,514 |
| Metadata duration | 461,084 seconds (128.0789 hours) |

Using the measured T4 reference throughput, the unoptimized full run is
projected to take approximately 249,800 seconds, or 69.39 hours. This is an
extrapolation, not an actual full-run timing, and must not be reported as an
observed result.

For a desktop RTX 5070, the current pipeline is provisionally estimated at
roughly 20–30 hours. This is also an estimate. Obtain a hardware-specific
estimate by running the stable reference fixture and applying:

```text
estimated full hours = reference wall seconds * 461084 / 315 / 3600
```

## Full-run decision

The full 50-shard baseline run is deferred. The current serial FP32 baseline
cannot finish within Kaggle's session window, and the local environment has no
available NVIDIA GPU. Preserve this baseline and its stable reference results.

If time and suitable hardware are available after implementation changes:

1. Run the preserved baseline on all 50 shards and record the actual wall time.
2. Run the final optimized version on the same hardware and input.
3. Validate both outputs and record hardware details independently.
4. Keep measured timings distinct from projections.
