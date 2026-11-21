# Worker Task: Phase 5 Factual Report and Finalist Correction

## Purpose

This is a narrow corrective follow-up to the completed Phase 5 T4 experiment.
Do **not** rerun the full 48-configuration screen. Preserve all existing raw
artifacts and measurements. Correct factual/reporting defects, add the one
missing required finalist, validate, and stop for Codex review.

Read `AGENTS.md`, `CODEX.md`, `OPTIMIZATION_EXPERIMENT_PLAN.md`,
`ai_chats/GEMINI_PHASE5_COMPREHENSIVE_OPTIMIZATION_RESULT.md`, and this file.
Do not modify `WRITEUP.md`, `.env`, reference artifacts, or prior Phase 4
benchmarks. Do not commit or expose credentials.

## Defects found in Codex review

1. `scripts/benchmark_configurations_driver.py` has hard-coded report claims
   that contradict `raw_results.csv`:
   - It claims FP16 batch 8 is fastest, while the 10-video screen identifies
     `cfg_33_D2_fp16_b4` at 72.65 FPS as the fastest FP16 and overall screen
     result.
   - It recommends D3 FP32 batch 2 as strict best, while the screen shows
     `cfg_26_D2_fp32_b2` at 45.52 FPS vs D3's 45.44 FPS.
   - It contains incompatible, stale decoder FPS values and unsupported claims
     that FP16 inference exceeded 110 FPS / end-to-end improved 2.4x.
2. `summary.csv` calculates finalist `speedup_vs_baseline` using the one-run
   screening baseline rather than the three-repeat finalist D0 FP32 b1 median.
   Its D0 baseline must be exactly `1.000x`; the existing finalists should use
   runtime ratio `baseline_median_runtime / candidate_median_runtime`, giving
   approximately 1.064x for D2 FP32 b2 and 1.756x for D2 FP16 b4.
3. The required finalist set omitted the best *threaded-decoding-only* D1
   configuration. Based on the screening table this is
   `cfg_21_D1_fp16_b4` (D1, fp16, batch 4, 68.05 FPS). It must receive three
   independent isolated full-ten-video repetitions before final ranking.
4. `scripts/compare_predictions.py` does not fully validate record schema,
   duplicate non-error keys, or contents of error records. Strengthen it and
   add focused tests. Its CLI exit behavior may remain nonzero for a mismatch,
   but the benchmark runner must record rather than abort on expected FP16
   differences.

## Required work

### A. Correct reporting source of truth

Make `benchmark_configurations_driver.py` derive all conclusions and report
numbers from actual raw/summary rows. Eliminate static performance claims.
Use clearly labelled source scopes:

- **screening**: one ten-video run per configuration;
- **finalist**: median of actual three independent ten-video runs;
- never compare a finalist median against a screening baseline.

If a value does not have a matching source, omit it rather than estimating it.
The regenerated `report.md` must consistently identify:

- screening fastest overall / FP16: `cfg_33_D2_fp16_b4`, 72.65 FPS;
- screening fastest strict FP32: `cfg_26_D2_fp32_b2`, 45.52 FPS;
- candidate selection vs final ranking must be labelled correctly;
- actual assignment accuracy: unverified, because no labeled ground truth.

Do not claim a configuration is compliant with the assignment based only on
agreement with the local FP32 reference.

### B. Run the missing finalist only

Use the existing private Kaggle optimization kernel/source and reference
dataset. Launch a new kernel version if needed and run only
`cfg_21_D1_fp16_b4` for three independent isolated full-ten-video repetitions,
with the same warmup/timing/telemetry/correctness methodology as the existing
finalists. Append its three raw run artifacts; do not rewrite any earlier raw
measurements. Download and validate outputs.

For each repetition, assert 10 videos, 7,473 decoded frames, 316 JSONL
records, zero unexpected errors, valid JSONL schema, and record the exact
reference-difference metrics. FP16 differences must be recorded, not treated
as a benchmark failure.

### C. Regenerate all affected deliverables

Regenerate from preserved raw data plus the new D1 finalist:

- `benchmarks/optimization/summary.csv`
- `benchmarks/optimization/report.md`
- plots whose content/ranking changes (at least `end_to_end_fps.png` and any
  finalist/ranking plot if one exists)
- `ai_chats/GEMINI_PHASE5_COMPREHENSIVE_OPTIMIZATION_RESULT.md`

The summary must now contain four finalists: D0 FP32 b1, D2 FP32 b2, D2 FP16
b4, D1 FP16 b4. Use actual medians and standard deviations only.

### D. Comparator correctness

Extend `compare_predictions.py` and tests to report, at a minimum:

- JSON parse/schema-invalid records;
- duplicate `(video, second)` non-error keys;
- missing/unexpected records;
- canonical object order/unknown class violations;
- comparison of error-record keys/content when errors occur.

Ensure malformed/duplicate/schema-invalid candidate output cannot be marked
`exact_match=True`. Preserve valid baseline and FP16 comparison behavior.

### E. Verify

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Reconfirm all frozen hashes. Write
`ai_chats/GEMINI_PHASE5_CORRECTIONS_RESULT.md` including the new Kaggle kernel
version/status, the three new raw paths/results, corrected summary table,
reference hashes, tests, files changed, and any limitations. Then stop. Do not
commit.
