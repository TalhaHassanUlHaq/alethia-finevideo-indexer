# Worker Task: Complete T4 FineVideo Optimization Experiment

## Authority and operating mode

You are the implementation worker. Execute this entire bounded task without
waiting for intermediate Codex feedback. Continue through every feasible T4
experiment, record failures safely, and stop only after the final completion
record is written. Codex will review tomorrow.

This task is T4-only. Do **not** create RTX 4090 tuning/portability tooling,
run RTX 4090 jobs, or make claims about 4090 performance. Avoid hard-coding
T4 memory or a T4-specific batch size in production code.

Read before editing:

1. `AGENTS.md`
2. `README.md`
3. `CODEX.md`
4. `OPTIMIZATION_EXPERIMENT_PLAN.md`
5. `ai_chats/GEMINI_PHASE4B_PROFILE_RESULT.md`
6. This file

Do not modify `WRITEUP.md`, `.env`, the immutable reference artifacts, or the
already-promoted Phase 4 batch/profile results. Never expose credentials. Do
not commit.

## Baseline invariants

The original command must remain completely valid and default to its prior
behavior:

```powershell
python run.py --input <folder> --output results.jsonl
```

Default means D0 serial PyAV, FP32, batch size 1, YOLO11l, image size 640,
confidence 0.4, class-aware NMS IoU 0.7, default Ultralytics letterbox, BGR
frames, current rotation behavior, and current aggregation.

Immutable reference facts:

- Fixture: `artifacts/reference/input/reference.parquet`
- Expected JSONL: `reference/expected.jsonl`
- 10 videos, 7,473 decoded frames, 316 JSONL records, zero errors
- Fixture SHA-256:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- Expected JSONL SHA-256:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- YOLO11l SHA-256:
  `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f`
- T4 reference dataset: `talhahassanulhaq/althia-reference-suite`

Validate hashes before and after work. Do not overwrite references or existing
`benchmarks/batch_size/` / `benchmarks/pipeline_profile/` content.

## Phase A: inspect and establish the measurement foundation

1. Inspect current pipeline, batching, profiling, and reference comparison
   code. Reuse it where practical rather than duplicating semantics.
2. Record exact versions of Python, PyTorch, CUDA, Ultralytics, PyAV,
   PyArrow, NumPy, GPU name/VRAM, logical CPU count, RAM, and CUDA runtime in
   a raw environment artifact and report.
3. Run the existing full test suite and compile command before implementation.
4. Add focused regression tests for every new behavior. Do not weaken any
   existing test.

## Phase B: backward-compatible configuration and production implementation

Create a small, typed configuration abstraction in `src/althia/`; keep it
minimal and clearly separated from benchmarking. Add optional CLI arguments
with these defaults:

- `--decoder-strategy D0` (choices exactly `D0`, `D1`, `D2`, `D3`)
- `--decode-thread-type AUTO` where D1/D3 use it (support `AUTO` and `FRAME`)
- `--decode-threads` (positive integer; default a documented deterministic
  setting chosen by Phase C, but D0 must not activate threaded decoding)
- `--prefetch-capacity` (positive bounded queue capacity; only D2/D3)
- `--precision fp32` (choices `fp32`, `fp16`)
- retain `--batch-size 1`

Thread/queue controls must have safe validation and must not alter D0 default
execution. Do not add automatic runtime tuning or GPU detection that changes
production output.

### D0: serial PyAV

Preserve the current decoder exactly as the compatibility reference.

### D1: multithreaded PyAV

Add a decoder configuration that sets only documented PyAV stream codec-context
thread settings (`thread_type` and `thread_count`) before decoding. Never share
a PyAV container/stream across threads. Unsupported settings must fail clearly
in benchmark setup rather than silently falling back while labelled D1.

### D2: CPU decode with bounded prefetch

Implement a per-video producer-consumer iterable. The producer owns the PyAV
decoder/container and performs decode, BGR conversion, and rotation. It emits
`DecodedFrame` values to a bounded `queue.Queue`; consumer owns inference and
aggregation. Preserve exact order/timestamps; process the final partial
inference batch; propagate producer exceptions after already-decoded frames;
terminate and join reliably; avoid deadlocks even when consumer inference
fails. Never retain an entire video in memory or cross video boundaries.

### D3: threaded PyAV plus prefetch

Combine D1's selected deterministic PyAV settings with D2. Use the same
semantics and shutdown/error behavior.

Do not add multi-video parallelism, concurrent GPU inference, frame skipping,
resizing changes, or a second CUDA stream.

### Precision

Implement FP16 through the installed Ultralytics prediction API only after
checking its installed signature/documentation. Pass an explicit `half`/the
supported precision argument for every inference call so FP32 remains
explicitly FP32. Do not mutate weights in place or alter model architecture.
If FP16 needs CUDA and CUDA is unavailable, fail early with a clear error. No
TF32 change.

## Phase C: decoder-only equivalence and setting selection

Create `scripts/benchmark_decoding.py`.

Using the ten-video fixture, run D0 and every supported D1 candidate:

- `AUTO`, threads 2
- `AUTO`, threads 4
- `FRAME`, threads 2
- `FRAME`, threads 4

No GPU inference is needed for this decoder-only probe. For every video and
candidate, compare against D0:

- video count and frame count
- frame order
- exact PTS/timestamp and time-base representation
- elapsed-second assignment
- rotation/final-frame behavior
- dimensions/dtype and a deterministic digest of post-rotation BGR pixels

Record decode wall time and FPS. Pick the fastest fully equivalent supported
setting as the deterministic D1/D3 benchmark setting; if within 5%, choose the
simpler/lower-thread choice. If candidates fail exact equivalence or are
unsupported, record and exclude them. Never claim D1 is active when it is not.

Also compare D2 and D3 decoded streams to D0 with the same evidence. Include
small fake-decoder tests for bounded queue, ordering, final partial batch,
producer exceptions, consumer exceptions, worker shutdown, and no deadlock.

## Phase D: prediction comparison and experiment infrastructure

Create these required files:

- `scripts/benchmark_configurations.py`
- `scripts/compare_predictions.py`
- `benchmarks/optimization/configurations.csv`
- `benchmarks/optimization/raw_results.csv`
- `benchmarks/optimization/summary.csv`
- `benchmarks/optimization/report.md`

Generate the full configuration matrix programmatically:

- decoder: `D0`, `D1`, `D2`, `D3`
- precision: `fp32`, `fp16`
- batch: `1`, `2`, `4`, `8`, `16`, `32`

This is 48 configurations. `configurations.csv` must include every requested
configuration and selected D1 settings, not merely successful configurations.

The benchmark runner needs driver/isolated-worker modes and must:

- use a fixed ten-video order and controlled seeds;
- warm up model/CUDA before steady-state timing;
- report initialization, warmup, steady-state, and cold-start wall time
  separately;
- synchronise CUDA only at meaningful timing boundaries, never after every
  frame; do not destroy intended prefetch overlap;
- record end-to-end and stage timing, decoded frames, inference calls,
  videos, JSONL records, errors, exception/OOM data, peak GPU VRAM, host RAM,
  CPU utilization, GPU utilization, decoder FPS, inference FPS;
- for D2/D3 record queue wait time, enqueue/dequeue blocking time, occupancy
  mean/max, capacity, producer/consumer completion state;
- distinguish overlapping worker accumulated time from end-to-end wall time;
- preserve raw per-run JSON/CSV/log/mismatch evidence in ignored `artifacts/`;
- continue safely after OOM/crash. Do not let a failed process corrupt other
  result rows.

Reuse the existing profiler/monitor where appropriate, but keep normal
production execution free of benchmark instrumentation overhead.

`scripts/compare_predictions.py` must compare JSONL semantic fields and write
detailed reports: video IDs, seconds, schema, canonical object order,
missing/unexpected records, objects, people, per-class mismatch counts,
object micro precision/recall/F1 *against the FP32 reference*, people exact
and within-one agreement. Identify confidence-related differences only when
confidence evidence was actually captured. Do not call reference agreement
ground-truth accuracy.

The official assignment thresholds are confirmed by the brief: object F1 >=
0.95, people exact >=85%, people within one >=95% **against the evaluator
reference**. The local fixture has no labeled ground truth, so mark actual
assignment accuracy unverified even if an FP16 configuration agrees with FP32.

## Phase E: execute the complete T4 experiment

Build a private Kaggle T4 kernel and stage only code plus the existing private
reference suite; create no replacement dataset and never include credentials.

1. Run one three-video feasibility worker for each of the 48 configurations.
   A failure, OOM, timeout, or mismatch is recorded; it does not stop the
   remainder.
2. Run one full ten-video screening worker for every configuration that is
   feasible on the three-video fixture. Exact-reference FP32 is a strict gate
   for strict-equivalence ranking. FP16 differences are expected to be
   measured and reported, never silently accepted.
3. From full-screen results choose up to six deduplicated finalists including,
   when feasible: FP32 D0 batch1 baseline, fastest FP32, fastest FP16, best
   prefetch, best threaded decode, and best overall.
4. Run every finalist three independent isolated repetitions in rotated order,
   provided the Kaggle session/time budget permits. If it does not, complete as
   many as possible, preserve all screen data, and state precisely what was not
   repeated and why. Do not fabricate median/variance data.

Order screening configurations in a rotated/interleaved order by decoder,
precision, and batch so thermal/load effects are not concentrated in one
factor. The original T4 FP32 D0 batch1 baseline must be included with the same
methodology for speedup computation.

Optional NVDEC investigation: only probe import/runtime availability and codec
compatibility without installing new dependencies or adding it to the matrix.
Document it as unavailable/unsupported unless it demonstrably preserves the
required in-memory PyAV semantics. It must never block the core experiment.

## Phase F: deliverables and report

Promote only required summaries and charts to `benchmarks/optimization/`:

- `configurations.csv`, `raw_results.csv`, `summary.csv`, `report.md`
- `end_to_end_fps.png`
- `throughput_heatmap.png`
- `memory_vs_throughput.png`
- `decode_vs_inference.png`
- `jsonl_differences.png`

Use real measurements only. Every raw row must have configuration ID, hardware
environment, decoder setting, precision, batch, counts, timings, utilization,
memory, queue fields, match metrics, accuracy-status label, status, and raw
artifact paths.

The report must answer, based on data:

1. fastest decoding strategy;
2. effect of PyAV threading on decoding;
3. effect of prefetch on end-to-end FPS;
4. FP16 inference/end-to-end change;
5. best FP16 batch size on T4;
6. strictly equivalent configurations;
7. configurations with actual assignment accuracy verified (likely none
   without ground truth);
8. best measured T4 configuration;
9. which configuration(s) should be validated first on RTX 4090, explicitly
   labelled unmeasured on that GPU;
10. limitations: ten videos, benchmark overhead, failed/skipped configurations,
    and any incomplete repetitions.

Rank separately: fastest T4, fastest strictly equivalent, fastest
accuracy-validated, and lowest-memory eligible. A configuration with
unverified FP16 assignment accuracy cannot be called conclusively compliant.
If candidates are within 5%, prefer lower memory and simpler implementation,
unless repetitions show a reliable difference.

## Required verification and completion record

Run after all code changes:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Write `ai_chats/GEMINI_PHASE5_COMPREHENSIVE_OPTIMIZATION_RESULT.md` with:

- files changed and why;
- before/after reference hashes;
- tests and exact pass count;
- D1 selection/equivalence evidence;
- Kaggle kernel URL, version, status and raw download path;
- all completed/skipped/failed configuration counts;
- exact locations of tables, reports, plots and raw artifacts;
- complete measured results and any OOM/errors;
- factual recommendations and remaining limitations.

Then stop for Codex review. Do not commit.
