# Implementation Task: T4 Batch-Size Experiment

## Role and scope

Gemini is the implementation and execution worker for this bounded experiment.
Codex owns the plan and final review. Implement, launch, monitor, and collect
the experiment exactly as specified here. Do not research alternative
optimizations or expand the scope.

## Immutable checkpoint and references

Start from Git commit:

`a6ebfc0fa9d4b8667faeba239ee4a5e31bb89ff8`

The working tree was clean before this task. Do not alter or replace these
authoritative artifacts:

| Artifact | Required SHA-256 |
|---|---|
| `reference/expected.jsonl` | `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` |
| `artifacts/reference/input/reference.parquet` | `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` |
| `yolo11l.pt` used on Kaggle | `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` |

Reference environment:

- NVIDIA T4, device 0
- PyTorch `2.8.0+cu128`
- torchvision `0.23.0+cu128`
- Ultralytics `8.4.216`
- PyAV `18.1.0`
- PyArrow `26.0.1`
- NumPy `2.2.6`
- FP32, image size 640, confidence 0.4, IoU 0.7
- 10 videos, 316 JSONL records, 8,507 decoded frames, zero errors

Before editing, verify the local hashes and record them. After all work, verify
again that both reference files are byte-for-byte unchanged.

## Existing batching behavior

The production code already contains the required backward-compatible design:

- CLI `--batch-size`, default `1`
- Per-video bounded `pending` frame buffer
- `detect_many()` sends a list to Ultralytics for batches greater than one
- Final incomplete batches are flushed
- Aggregation and error isolation are unchanged

Inspect it before editing. Do not rewrite working production code merely to
claim implementation. Make only a confirmed minimal correction if a regression
test exposes a defect.

## Preprocessing equivalence condition

Pinned Ultralytics preprocessing selects minimum-rectangle letterboxing only
when all images passed to a prediction call have the same shape. The production
pipeline batches within one video, so frames are expected to have a consistent
shape, including the rotation-tagged fixture video after rotation.

Instrument every measured inference call with:

- Batch length
- Set of input image shapes
- Count of mixed-shape inference calls

Do not change `rect`, `LetterBox`, padding, image size, or any preprocessing
setting. If any measured call has mixed shapes, report it explicitly and
explain that Ultralytics may switch from minimum-rectangle to fixed-square
padding. Correctness comparison remains authoritative.

## Required implementation

### 1. Regression coverage

Add focused tests proving:

- Default CLI batch size remains 1.
- For a configurable batch size, decoded frames retain exact order and
  timestamps.
- Full batches and the final incomplete batch are processed once each.
- A batch never crosses a video boundary.
- Batch-size validation rejects values below one.
- Batched detector results remain paired with the correct frames.
- Existing per-video partial-failure behavior remains intact with a batch size
  greater than one.

Use fakes only; unit tests must not load YOLO or require a GPU.

### 2. Benchmark program

Create `scripts/benchmark_batch_size.py`. It must support a driver mode and an
isolated worker mode so every measured run initializes its own process.

The driver must test progressive sizes `[1, 2, 4, 8, 16, 32]` with three
independent measured repetitions for each non-OOM size. Use execution order:

- Repetition 1: ascending
- Repetition 2: descending over sizes that survived repetition 1
- Repetition 3: ascending

If a size OOMs during repetition 1, record it and do not test larger sizes.
Do not require three repetitions for an OOM size.

Test 64 only when batch 32:

- completes all three repetitions,
- exactly matches the reference in every repetition,
- retains at least 20% actual device-memory headroom at peak, and
- improves median end-to-end FPS over batch 16 by more than 5%.

Document the reason whether 64 is tested or skipped.

### 3. Isolated worker measurement

Each worker must:

1. Verify fixture, reference, model, source, settings, and package versions.
2. Load the model on T4 device 0 in FP32.
3. Warm up CUDA and the model using the candidate batch size before timing.
4. Reset counters and peak-memory statistics after warmup.
5. Start timing immediately before `run_pipeline` and stop after output is
   flushed; model initialization and warmup must be excluded.
6. Call `torch.cuda.synchronize()` around measured detector calls and at the
   full-run timing boundaries.
7. Exercise the production `UltralyticsDetector.detect_many()` path. You may
   wrap `model.predict` for instrumentation, but do not duplicate or replace
   production preprocessing/inference logic.
8. Capture Ultralytics `Results.speed["inference"]` totals for model
   inference-only time. Also record synchronized detector-call wall time.
9. Count decoded/inferred frames, inference calls, observed batch lengths,
   mixed-shape calls, completed videos, output records, and errors.
10. Require exactly 8,507 inferred frames, 10 videos, and 316 records for a
    successful run.
11. Validate the JSONL and compare it with the immutable reference using exact
    semantic comparison.
12. Write a detailed mismatch JSON containing missing/unexpected keys and every
    differing `objects` or `people` field. Never overwrite the reference.

### 4. GPU memory and OOM safety

During the timed run, sample actual device free/total memory using
`torch.cuda.mem_get_info()` at a short interval in a background sampler. Also
record PyTorch peak allocated and reserved memory.

Report:

- Actual peak device-used MiB
- Total device MiB
- Minimum free MiB
- Headroom fraction at peak
- Peak PyTorch allocated/reserved MiB

Eligibility requires at least 20% actual device-memory headroom at peak.

Treat any CUDA OOM message, output error caused by OOM, worker abnormal exit,
or missing metrics as an OOM/failed run. Preserve earlier outputs and logs. A
worker process that OOMs must terminate before the driver continues. Do not
test larger sizes after the first OOM without stopping and reporting why.

### 5. Metrics and selection

For every batch size, aggregate the median across successful repetitions:

- End-to-end runtime and FPS
- Ultralytics inference-only FPS
- Synchronized detector-call FPS
- Speedup relative to the fresh batch-1 median
- Peak actual device memory and memory headroom
- Peak allocated/reserved memory
- Decoded frames and inference calls
- Completed videos and JSONL records
- Mismatched records
- Errors and OOM failures

Definitions:

```text
end_to_end_fps = decoded_frames / end_to_end_runtime
inference_only_fps = decoded_frames / summed_ultralytics_inference_seconds
speedup = batch_1_median_runtime / candidate_median_runtime
```

A batch is strictly eligible only if all three repetitions succeed, process all
expected data, contain no mixed/duplicate/skipped frame evidence, match the
reference exactly, have no OOM/errors, and retain at least 20% actual memory
headroom.

Report separately:

- Fastest measured batch size
- Best strictly equivalent batch size
- Recommended production batch size

Choose the eligible size with highest median end-to-end FPS. If another
eligible smaller size is within 5% of the fastest eligible FPS, recommend the
smaller size.

### 6. Deliverables

Create:

- `scripts/benchmark_batch_size.py`
- `benchmarks/batch_size/results.csv`
- `benchmarks/batch_size/report.md`
- `benchmarks/batch_size/throughput.png`
- Regression tests for batching

Keep per-repetition JSONL files, detailed mismatch JSON, subprocess logs, and
raw metrics under ignored `artifacts/batch_size/`.

The report table must contain:

| Batch | End-to-end FPS | Speedup | Peak GPU memory | JSONL match | Status |
|---|---:|---:|---:|---|---|

The report must answer all questions in the user's experiment request using
actual measurements only. If a value was not measured, say `not measured`.

## Kaggle execution

Use the existing private resources only; do not create or replace a dataset:

- Dataset source: `talhahassanulhaq/althia-reference-suite`
- Kernel source: `talhahassanulhaq/althia-phase-3-reference`

The attached dataset contains the exact fixture. The attached kernel output
contains the exact expected JSONL, probe, metadata, source snapshot, and model
weight. Locate them by the required hashes, not by silently accepting the first
matching filename.

Create a private benchmark kernel configuration/source as needed. It must use
one NVIDIA T4 and may have internet enabled only for installing the pinned
environment. Do not upload a new dataset or modify reference artifacts.

Launch the benchmark kernel, monitor it to a terminal state, download outputs,
and promote only the aggregate CSV, report, and PNG into
`benchmarks/batch_size/`. Gemini is responsible for monitoring the long run per
`AGENTS.md`; Codex should not poll it.

## Constraints

- Do not enable FP16, TF32 explicitly, frame skipping, alternate resizing,
  concurrent decoding, multi-GPU, cross-video batching, or preprocessing
  changes.
- Do not change model weights, image size, confidence, IoU, NMS behavior,
  rotation, BGR input, aggregation, or output schema.
- Do not edit `WRITEUP.md`, `.env`, reference files, Phase 2/3 evidence, or raw
  FineVideo data.
- Never expose credentials.
- Do not commit changes; Codex will review and commit after the experiment.
- Never fabricate measurements.

## Completion record

Write `ai_chats/GEMINI_PHASE4_BATCH_RESULT.md` containing:

- Files changed
- Unit-test results
- Reference hashes before and after
- Kaggle kernel URL/version and terminal status
- Raw output locations
- Actual results table
- Fastest, best-equivalent, and recommended batch sizes
- Any OOM, mismatch, mixed-shape, memory, or measurement limitations

Then stop and return control to Codex for review.
