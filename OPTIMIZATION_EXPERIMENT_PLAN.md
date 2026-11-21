# FineVideo Optimization Experiment Plan

## Scope and guardrails

This is a controlled T4 experiment over decoding strategy, inference
precision, and batch size. The existing FP32 serial, batch-size-1 command is
the immutable compatibility baseline:

```powershell
python run.py --input <folder> --output results.jsonl
```

No experiment may skip frames, change timestamps, rotation, BGR conversion,
model weights, image size, thresholds, target classes, NMS, or aggregation.
`reference/expected.jsonl` and existing benchmark artifacts are immutable.

The T4 profiling evidence makes GPU inference the first optimization target:
batch-1 FP32 used 152.23 s (84.2% of 180.74 s) in inference, while decoding
and conversion used 10.88 s (6.0%). This motivates FP16 screening, but does
not establish FP16 correctness or forecast RTX 4090 results.

## Experiment sequence

1. **Freeze and verify**: validate reference/model hashes, record package/GPU/
   CPU/CUDA environment, inspect preprocessing, and run the complete test
   suite.
2. **Configuration and decoders**: add backward-compatible optional controls
   for D0 serial PyAV, D1 threaded PyAV, D2 bounded decode prefetch, D3
   threaded PyAV plus prefetch, FP32/FP16, and batch size.
3. **Decode-only gate**: test PyAV `AUTO` and `FRAME` with thread counts 2 and
   4. Select one documented D1 setting only after exact frame/timestamp/pixel
   equivalence against D0; record unsupported modes rather than guessing.
4. **Three-video feasibility gate**: run every generated D0--D3 × FP32/FP16 ×
   batch 1/2/4/8/16/32 configuration in isolated workers. Record failures and
   continue. FP16 is compared with FP32 reference; it is not assumed exact.
5. **Ten-video T4 screen**: run every feasible configuration once, keeping raw
   per-run data, stage timings, memory/utilization, queue evidence, JSONL
   comparisons, and errors.
6. **Finalists**: choose up to six deduplicated candidates (baseline, fastest
   FP32, fastest FP16, fastest prefetch, fastest threaded decode, overall
   fastest), then run each three times where Kaggle runtime permits. Rank by
   median end-to-end FPS with correctness and memory gates first.
7. **Review/report**: publish raw and aggregate tables, required plots, exact
   mismatch reports, limitations, and clear T4-only recommendations.

## Correctness standards

- D0/D1/D2/D3 at FP32 must exactly match the existing ten-video JSONL and must
  demonstrate identical decoded frame count, frame order, PTS/time base,
  elapsed-second assignments, rotation and final-frame behavior. Pixel digests
  are required where the decoder can be compared directly.
- FP16 must receive a detailed exact-reference comparison: differing records,
  percentage exact, object/people/per-class differences, and any retained
  confidence diagnostics. It is only eligible as strictly equivalent if the
  semantic output matches exactly.
- The official brief requires object F1 >= 0.95, people exact accuracy >= 85%,
  and people within-one accuracy >= 95% against evaluator reference results.
  Local FP32 output is a model reference, not labeled ground truth; actual
  assignment accuracy remains unverified unless ground truth is available.

## Explicitly deferred

RTX 4090-specific tuning/automatic tuning and an RTX 4090 portability script
are deferred. The implementation must nevertheless avoid hard-coding T4 VRAM
or device-specific batch choices, so later validation remains possible.
