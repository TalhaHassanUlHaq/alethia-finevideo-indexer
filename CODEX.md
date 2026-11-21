# Codex Handoff

## Role boundary

- Codex is the orchestrator, planner, supervisor, reviewer, and researcher.
- Gemini / Antigravity is the implementation worker only.
- Give Gemini bounded tasks through `ai_chats/GEMINI_CONTEXT.md` and a task
  file. Gemini must not independently choose research, architecture, or
  optimization direction.
- For long Kaggle runs, Gemini monitors until completion and reports through a
  completion record. Codex should not continuously poll a running job.
- Do not edit `WRITEUP.md`; it is solely user-authored.

## Repository and checkpoint

- Workspace: `F:\Lums courses\FALL2026\side_projects\Althia`
- Initial accepted checkpoint:
  `a6ebfc0fa9d4b8667faeba239ee4a5e31bb89ff8`
  (`Complete accurate baseline and reference suite`)
- Phase 4 batch-size work and the current profiling task are uncommitted and
  require Codex review before any commit.
- Preserve unrelated user changes in a dirty worktree. Use `apply_patch` for
  edits. Git operations may require escalation because of sandbox refresh.

## Assignment status

### Completed and accepted

1. Correctness-first six-stage baseline.
2. Stable 10-video local/Kaggle reference suite.

Reference invariants:

- Input fixture: `artifacts/reference/input/reference.parquet`
- Expected output: `reference/expected.jsonl`
- Fixture SHA-256:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- Expected JSONL SHA-256:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- YOLO11l weights SHA-256:
  `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f`
- Correct expected counts: 10 videos, 7,473 decoded frames, 316 JSONL records,
  zero errors.
- Dataset: `talhahassanulhaq/althia-reference-suite`
- Reference Kaggle kernel source: `talhahassanulhaq/althia-phase-3-reference`
- T4 reference version 3: 170.656 seconds, exact output.

The fixture intentionally covers people entering/leaving, short appearances,
scene changes, crowds, unusual frame rates, orientation metadata, partial
final seconds, and damaged-video behavior.

### Phase 4 batch-size experiment: Gemini completed; pending review/commit

Objective: configurable batched YOLO11l FP32 inference, exact semantic
equivalence to the immutable reference, and T4 performance measurements.

Known Gemini-reported results, each three runs on the 10-video fixture:

| Batch | Median runtime (s) | E2E FPS | Inference FPS | Speedup | Peak GPU MiB | Exact JSONL |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 176.0243 | 42.45 | 51.25 | 1.000x | 412.9 | Yes |
| 2 | 171.5872 | 43.55 | 51.86 | 1.026x | 542.9 | Yes |
| 4 | 179.8470 | 41.55 | 48.85 | 0.979x | 840.9 | Yes |
| 8 | 186.7828 | 40.01 | 46.43 | 0.942x | 1226.9 | Yes |
| 16 | 186.0540 | 40.17 | 46.37 | 0.946x | 2184.9 | Yes |
| 32 | 195.3535 | 38.25 | 43.82 | 0.901x | 4092.9 | Yes |

- All six sizes exact; no OOM; no mixed-shape batches.
- Batch 64 intentionally skipped because batch 32 regressed versus 16.
- Fastest measured / best strictly equivalent: batch 2.
- Production recommendation under the stated 5% tie rule: batch 1.
- Kernel: `talhahassanulhaq/althia-batch-size-benchmark`, version 1,
  reported COMPLETE.

Files expected from that work:

- `scripts/benchmark_batch_size.py`
- `scripts/build_kaggle_benchmark.py`
- `benchmarks/batch_size/`
- `kaggle/batch_size/`
- `ai_chats/GEMINI_PHASE4_BATCH_RESULT.md`

Review warning: the earlier batch report claimed CPU video decoding was the
dominant bottleneck. Its own batch-1 data instead reports roughly 176.0 s
end-to-end and 163.4 s in synchronized detector calls, so that claim is not
supported. Do not repeat it without the profiling evidence below.

## Current active task: comprehensive T4 optimization experiment

Gemini task: `ai_chats/GEMINI_TASK_PHASE5_COMPREHENSIVE_OPTIMIZATION.md`

Context pointer: `ai_chats/GEMINI_CONTEXT.md`

The Phase 4B profiling task is complete. Its empirical result is that T4 FP32
batch-1 is GPU-compute bound: 152.23 s of 180.74 s (84.2%) was GPU inference,
whereas PyAV decoding plus conversion was 10.88 s (6.0%). All four profile
outputs exactly matched the immutable reference.

The current task is one complete overnight T4-only experiment, covering D0
serial PyAV, D1 threaded PyAV, D2 bounded decode prefetch, D3 threaded PyAV
plus prefetch, FP32/FP16, and batch sizes 1/2/4/8/16/32. It must complete
decoder-only equivalence, 48 three-video feasibility runs, every feasible
ten-video screen, and up to six three-run finalists. It records failures and
continues; it must never fabricate results.

The task deliberately defers RTX 4090-specific tuning and portability tooling.
No production code may hard-code T4 VRAM/batch decisions, however.

Required new deliverables are listed in the task and include:

- `scripts/benchmark_decoding.py`
- `scripts/benchmark_configurations.py`
- `scripts/compare_predictions.py`
- `benchmarks/optimization/` raw/summary/configuration CSVs, report, and five
  plots
- regression tests
- `ai_chats/GEMINI_PHASE5_COMPREHENSIVE_OPTIMIZATION_RESULT.md`

Use this prompt for Gemini if the task has not begun:

> Read `AGENTS.md`, `CODEX.md`, and `ai_chats/GEMINI_CONTEXT.md` completely.
> Execute the active Phase 5 comprehensive T4 optimization task in its
> entirety. Work through all feasible experiments while I am away; safely
> record failures or incomplete runs, validate outputs, write the completion
> record, and stop only for Codex review. Do not modify `WRITEUP.md`, access
> `.env`, commit, or build RTX 4090-specific tooling.

## After Gemini reports completion

1. Inspect `ai_chats/GEMINI_PHASE5_COMPREHENSIVE_OPTIMIZATION_RESULT.md` and
   every changed file.
2. Validate the reported hashes, frame/video/record counts, exact JSONL match,
   GPU sampling coverage, accounting equation, and timing units.
3. Run the focused tests plus the full test suite locally where feasible.
4. Check that production code behavior remains backward compatible and that
   instrumentation is benchmark-only/no-op by default.
5. Verify strict decoding equivalence and ensure FP16 is clearly labelled
   reference-agreement only, not ground-truth assignment accuracy.
6. Check all 48 configurations/failures are represented in raw results, and
   repeat-run statistics are only reported for actual repetitions.
7. Make recommendations based on the measured T4 data; keep RTX 4090 claims
   explicitly unmeasured.
8. Commit only accepted work in a focused commit after review.

## Useful project notes

- Full source dataset: 50 shards, 1,650 videos, 24.82 GB, 461,084 seconds.
- A later frame-based estimate used 12,984,759 full-dataset frames versus 7,473
  reference frames (ratio 1737.56), projecting about 82.37 hours from the
  170.656-second T4 reference run. Earlier duration-based projections in
  `BENCHMARKS.md` are labeled estimates and differ; do not silently present
  either as an actual full-dataset measurement.
- The current production baseline uses YOLO11l, FP32, image size 640,
  confidence 0.4, NMS IoU 0.7, PyAV decoding every frame, and canonical
  five-class aggregation.
- Per-second object presence threshold remains
  `ceil(frames_in_second / 2)`; people count remains floored median over all
  frames in each second. These must never change in a batch/profiling task.

## Superseding status update

Phase 5 completed on the T4 and was followed by a no-inference release-prep
pass. The CPU-safe test suite now has 53 passing tests. The report generator
and JSONL comparator were corrected to avoid the earlier hard-coded ranking
claims; current benchmark facts are in `benchmarks/optimization/`.

The selected future production command, requested by the user, is:

```text
python run.py --input <folder> --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 16
```

No full dataset or local GPU is available in this workspace, and the user has
explicitly instructed us not to run inference here. The repository is
structured for transfer to a GPU-equipped friend through `RUN_ON_FRIEND.md`.
The actual full-run output, full-run hardware facts, and user-authored writeup
remain pending:

- `results.jsonl.gz` must be generated by the real 50-shard run and committed
  at repository root. `.gitignore` explicitly permits this exact file.
- `HARDWARE.md` is a factual template pending that run.
- `WRITEUP.md` must be authored by the user without AI help.
- `ai_chats/links.txt` is a placeholder for user-exported complete Codex and
  Gemini histories; add them before public submission.

No active Gemini task remains. Do not launch any model, Kaggle, benchmark, or
dataset download unless the user explicitly reverses the no-inference
instruction.
