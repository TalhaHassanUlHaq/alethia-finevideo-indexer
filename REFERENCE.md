# Reference Test Set

This directory and guide provide the reproducible local reference suite used to
verify every optimization against the correctness-first baseline.

## Tracked vs Ignored Files

- **Tracked files** (committed to version control):
  - `reference/candidates.json`: Frozen list of 10 source video coordinates and rationale.
  - `reference/expected.jsonl`: Stable reference answers produced by FP32 baseline on reference fixture.
  - `reference/metadata.json`: Provenance metadata (fixture, result, model, unique source coordinate hashes, per-fixture-row provenance with transforms, package versions, timing).
  - `scripts/build_reference_fixture.py`: Script to build deterministic reference parquet.
  - `scripts/probe_reference_fixture.py`: PyAV fixture probe for frame counts, dimensions, rates, and rotation.
  - `scripts/generate_reference.py`: Reference generator invoking production baseline.
  - `scripts/compare_reference.py`: Comparator reporting F1, people match rates, exact equality, and schema validity.
  - `tests/test_reference.py`: Unit tests for reference infrastructure.

- **Ignored files** (under `artifacts/reference/`, never committed):
  - `artifacts/reference/input/reference.parquet`: Compact 10-video fixture Parquet.
  - `artifacts/reference/input/reference-manifest.json`: Fixture manifest with per-video SHA-256 digests.
  - `artifacts/reference/probe.json`: Probe output containing decoded frame counts, timestamps, and rotation tags.

## Frozen Candidate Selection

The reference set comprises 10 videos representing critical coverage dimensions:

| Fixture Row | Source Shard | Source Row | Duration | Primary Coverage |
|---:|---|---:|---:|---|
| 0 | `train-00000-of-01357.parquet` | 5 | 35 s | Crowded scene; baseline observed up to five people |
| 1 | `train-00034-of-01357.parquet` | 14 | 22 s | Red-carpet people movement, brief appearances, portrait footage |
| 2 | `train-00032-of-01357.parquet` | 21 | 20 s | Racing replay, short car appearances and scene changes |
| 3 | `train-00043-of-01357.parquet` | 9 | 21 s | Sports highlights and rapid scene changes |
| 4 | `train-00040-of-01357.parquet` | 18 | 23 s | Group/crowd movement and scene changes |
| 5 | `train-00021-of-01357.parquet` | 22 | 75 s | Unusual 10 FPS source |
| 6 | `train-00023-of-01357.parquet` | 11 | 31 s | Dog and portrait footage |
| 7 | `train-00045-of-01357.parquet` | 24 | 31 s | Laptop footage |
| 8 | `train-00037-of-01357.parquet` | 26 | 22 s | Cell-phone footage and portrait orientation |
| 9 | `train-00000-of-01357.parquet` | 5 | 35 s | Deterministic nonzero rotation-metadata edge case (90° display rotation transform) |

Total duration: 315 seconds across 10 videos.

## Workflow Commands

### 1. Build Reference Fixture

Extracts only the selected rows from local FineVideo shards into a compact Parquet fixture:

```bash
python scripts/build_reference_fixture.py \
  --shards-dir finevideo/data \
  --output-dir artifacts/reference/input \
  --selection reference/candidates.json
```

Add `--force` to overwrite existing fixture artifacts if candidates change.

### 2. Probe Fixture Media

Probes video streams and decoded frames with PyAV:

```bash
python scripts/probe_reference_fixture.py \
  --fixture artifacts/reference/input/reference.parquet \
  --output artifacts/reference/probe.json
```

Outputs decoded frame count, timestamps, dimensions, stream rate, and rotation tags.

### 3. Generate Reference Answers (GPU)

Runs the production baseline (FP32, batch size 1, all frames, default YOLO11l settings) to produce stable reference answers:

```bash
python scripts/generate_reference.py \
  --fixture-dir artifacts/reference/input \
  --output-jsonl reference/expected.jsonl \
  --output-metadata reference/metadata.json \
  --model yolo11l.pt \
  --device 0
```

### 4. Validate Reference JSONL

Validates the output against assignment constraints:

```bash
python scripts/validate_jsonl.py reference/expected.jsonl --expected-videos 10 --no-errors
```

### 5. Compare Candidate Against Reference

Compares candidate results (e.g., from an optimized pipeline) against `reference/expected.jsonl`:

```bash
# Default threshold mode: exits non-zero if F1 < 0.95, people exact < 85%, or within-one < 95%
python scripts/compare_reference.py results-candidate.jsonl

# Exact equivalence mode: exits non-zero on any difference
python scripts/compare_reference.py results-candidate.jsonl --exact

# JSON report output
python scripts/compare_reference.py results-candidate.jsonl --json
```

## Comparison Metrics and Acceptance Thresholds

The comparator evaluates:
1. **Record Key Coverage**: All `(video, second)` records must be accounted for.
2. **Object Micro F1**: Must be `>= 0.95` (harmonic mean of multi-label precision and recall).
3. **People Exact Match Rate**: Must be `>= 85.0%` (`candidate people == reference people`).
4. **People Within-One Rate**: Must be `>= 95.0%` (`|candidate people - reference people| <= 1`).
5. **Exact Record Match Rate**: Rate of records where both objects and people match bit-for-bit.
6. **Schema Validity**: Schema failures immediately fail the comparison.

## Verified Phase 3 Reference

Phase 3 is complete. The authoritative reference was generated by private
Kaggle kernel version 3 using the all-frame FP32, batch-size-1 baseline.

| Measurement | Verified value |
|---|---:|
| Fixture videos | 10 |
| Output seconds | 316 |
| Error records | 0 |
| Kaggle elapsed time | 170.656215052 seconds |
| Rotation metadata case | Fixture row 9, 90 degrees |
| Unit tests | 29 passed |

Hashes:

- Fixture SHA-256:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- YOLO11l weight SHA-256:
  `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f`
- Reference result SHA-256:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`

`reference/expected.jsonl` passes the project validator for 10 videos and exact
self-comparison. `reference/metadata.json` records per-fixture-row source and
derived payload provenance without collisions.
