# Worker Task: T4 Pipeline Timing and Utilization Profile

## Objective

Measure where the current all-frame FP32 pipeline spends time for batch sizes
`1`, `2`, `8`, and `32`, and determine from evidence whether the primary limit
is CPU decode/conversion, YOLO preprocessing/data transfer, GPU inference,
postprocessing, aggregation/output, or synchronization/framework overhead.

This is a measurement-only follow-up. Do not change production behavior or
introduce FP16, concurrency, frame skipping, preprocessing changes, or any
other optimization.

## Preserve existing work

Do not overwrite or edit:

- `reference/expected.jsonl`
- `reference/metadata.json`
- `benchmarks/batch_size/results.csv`
- `benchmarks/batch_size/report.md`
- `benchmarks/batch_size/throughput.png`
- Existing raw batch-size artifacts

Required reference hashes remain:

- Fixture:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- Expected JSONL:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- YOLO11l weight:
  `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f`

The correct fixture contains 10 videos, 7,473 decoded frames, and 316 JSONL
records. Verify these values and all hashes before and after profiling.

## Important open question

The batch-size report currently claims CPU decoding is dominant, but its own
batch-1 medians show approximately 176.0 seconds end-to-end versus 163.4
seconds inside synchronized detector calls. That claim is not established.
Treat the bottleneck as unknown until this profile is complete. Correct the
batch-size report later only if the new measured profile justifies it; do not
silently rewrite the prior measured tables.

## Profiling design

Create `scripts/profile_pipeline.py` with driver and isolated worker modes.
Reuse the existing production pipeline, detector, fixture, weights, and exact
reference. Instrument through wrappers/monkeypatches or backward-compatible
no-op hooks; do not duplicate detection, preprocessing, aggregation, or output
semantics.

Run one independent profiled repetition per size using isolated processes, in
this order: `1, 2, 8, 32`.

This reduced run count is an explicit time-saving decision. Do not claim that
one observation establishes repeat-run timing variability or statistical
stability. Preserve the raw per-run data so additional repetitions can be
appended later without rerunning or overwriting this experiment.

Initialize and warm up the model before each measured run. Exclude
model initialization and warmup. Synchronize CUDA at full-run timing
boundaries. Every output must exactly match the immutable reference.

Profiling runs are separate from the existing throughput results. Do not use
their instrumented wall times to replace the original batch-size benchmark.

## Required time categories

Record seconds and percentage of end-to-end wall time for each batch size's
single measured run:

1. **Parquet loading**
   - Opening/iterating Parquet and materializing each `mp4` payload.
2. **Video decoding**
   - Container open and time spent obtaining decoded PyAV frames, excluding
     ndarray conversion/rotation.
3. **Rotation and frame conversion**
   - Timestamp handling, `frame.to_ndarray(format="bgr24")`, rotation, and
     contiguous-array conversion.
4. **YOLO preprocessing**
   - Use summed Ultralytics `Results.speed["preprocess"]`; this includes
     letterbox/tensor preparation and host-to-device preparation reported by
     Ultralytics. Do not alter preprocessing to isolate transfer.
5. **GPU inference**
   - Use summed Ultralytics `Results.speed["inference"]` and synchronized
     detector-call wall time as separate fields.
6. **YOLO postprocessing**
   - Use summed Ultralytics `Results.speed["postprocess"]`, including NMS.
7. **Aggregation and JSONL writing**
   - Measure aggregation `add_frame`/record construction and JSON
     serialization/write time. Report aggregation and writing separately plus
     their combined total.
8. **Residual/framework/synchronization time**
   - Report the nonnegative remainder between end-to-end wall time and the
     explicitly attributed non-overlapping wall categories.
   - Also report detector overhead as synchronized detector-call wall time
     minus Ultralytics preprocess + inference + postprocess totals.

Because nested timers can overlap, provide a documented accounting equation.
Do not add overlapping nested values together and claim they equal total time.
If Ultralytics stage timers and outer wall timers do not reconcile, report the
difference rather than forcing it to zero.

## Instrumentation guidance

- Measure Parquet loading by wrapping iteration around
  `iter_parquet_videos()`.
- Use a benchmark-only profiled decoder matching `decode_video()` exactly:
  time container/decode iteration separately from timestamp/BGR/rotation work.
  Inject it into the existing `process_video` path without changing default
  production behavior.
- Time `VideoAggregation.add_frame`, record construction, and
  `write_json_line` through wrappers while preserving outputs.
- Wrap the existing detector's `model.predict` to capture all returned
  `Result.speed` dictionaries, input batch lengths, and shape sets. Continue
  exercising `UltralyticsDetector.detect_many()` itself.
- Record mixed-shape calls. Do not change letterboxing even if one occurs.
- Keep detailed per-call timing only under ignored raw-artifact paths to avoid
  bloated tracked files.

## GPU utilization and memory

During each timed run, use a low-overhead persistent NVIDIA monitoring process
or NVML sampler. Do not repeatedly spawn `nvidia-smi` in the hot path.

Collect at a documented interval no slower than one sample per second:

- GPU SM utilization: mean, median, p95, maximum
- Fraction of samples at or above 90% utilization
- Memory-controller utilization: mean and p95 when available
- Device memory used/free/total and peak used
- Power draw mean and p95 when available
- Sample count and interval

Also retain PyTorch peak allocated and reserved memory. Store raw utilization
samples under ignored `artifacts/pipeline_profile/`.

## CPU and variability evidence

For every run, record:

- Process user CPU seconds
- Process system CPU seconds
- Total process CPU seconds
- Host logical CPU count
- CPU-seconds / wall-second ratio
- End-to-end runtime

For each batch size, report the observed runtime and average GPU utilization
from its single run. Mark minimum, maximum, mean, median, standard deviation,
coefficient of variation, range percentage, and between-run GPU-utilization
variability as `N/A (one run)` rather than fabricating variability statistics.

## Correctness and counts

Each profiled run is valid only when it has:

- 10 completed videos
- 7,473 inferred frames
- 316 JSONL records
- Zero error records
- Zero mixed-shape calls, or an explicit warning if not zero
- Exact semantic equality with `reference/expected.jsonl`
- No missing or duplicate frame evidence

Write per-run mismatch reports under ignored artifacts. Do not overwrite the
reference.

## Diagnosis rules

Base the final diagnosis on measured fractions and utilization:

- High GPU-inference fraction plus sustained high GPU utilization indicates a
  GPU-compute bottleneck.
- High decode/conversion fraction with low or bursty GPU utilization indicates
  CPU/feeding bottleneck.
- High preprocessing fraction with low inference fraction suggests
  preprocessing/data-transfer pressure; do not claim transfer alone because
  Ultralytics combines preprocessing operations.
- High detector residual or end-to-end residual indicates framework,
  synchronization, or uninstrumented overhead requiring further investigation.
- A large postprocessing fraction indicates CPU/NMS pressure.

Do not infer a CPU bottleneck merely because larger batches are slower. State
measurement limitations and distinguish correlation from proof.

## Deliverables

Create:

- `scripts/profile_pipeline.py`
- `benchmarks/pipeline_profile/results.csv`
- `benchmarks/pipeline_profile/report.md`
- `benchmarks/pipeline_profile/timing_breakdown.png`
- `benchmarks/pipeline_profile/gpu_utilization.png`
- Focused unit tests for profiling/accounting helpers using fakes only
- Kaggle staging/kernel files needed for this profile

The report must include:

1. Median stage breakdown table for batches 1, 2, 8, 32.
2. GPU utilization and memory table.
3. Single-run timing table, with repeat-run variability explicitly marked as
   not measured.
4. Accounting/residual table.
5. Explicit answer: CPU, GPU, preprocessing/data transfer, postprocessing, or
   synchronization/framework bottleneck.
6. Evidence-based implications for FP16 versus decode/inference overlap,
   without implementing either.
7. Limitations of the ten-video fixture and one-second utilization sampling.

Never fabricate missing measurements. Use `not measured` where necessary.

## Kaggle execution and monitoring

Use one private T4 kernel and the same existing resources:

- Dataset: `talhahassanulhaq/althia-reference-suite`
- Kernel source: `talhahassanulhaq/althia-phase-3-reference`

Do not create or modify a dataset. Launch the profiling kernel, monitor it to a
terminal state, download raw outputs, validate them, and promote only the CSV,
report, and two PNGs. Gemini is responsible for monitoring the long run.

## Constraints

- No FP16, TF32 changes, frame skipping, cross-video batching, concurrent
  decoding, multi-GPU, model changes, threshold changes, resize changes, or
  aggregation changes.
- Do not edit `WRITEUP.md`, `.env`, reference artifacts, or existing benchmark
  measurements.
- Do not commit. Codex will review and commit accepted work.
- Never expose credentials.

## Completion record

Write `ai_chats/GEMINI_PHASE4B_PROFILE_RESULT.md` with files changed, test
results, kernel URL/version/status, raw paths, all measured tables, reference
hash verification, and the bottleneck diagnosis. Then stop for Codex review.
