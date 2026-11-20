# Implementation Task: Phase 3 Rotation-Metadata Fixture

## Context

The reviewed nine-video FP32 Kaggle run succeeded (281 output seconds, zero
errors, 142.003 seconds), but its real probe found no nonzero rotation tag.
Codex then scanned all 213 portrait videos in the local 50-shard dataset; all
had rotation 0 because FineVideo stores them physically upright.

Codex verified locally that a lossless PyAV remux works:

```python
output_stream = output.add_stream_from_template(input_stream)
output_stream.set_display_rotation(90)
```

After remuxing, PyAV decodes the frame with `frame.rotation == 90` while the
encoded dimensions remain unchanged. Implement a deterministic derived fixture
row using this mechanism.

## Required changes

### 1. Add a tenth frozen candidate

Append a derived candidate to `reference/candidates.json` using the same source
as fixture row 0:

- Source: `train-00000-of-01357.parquet`, row 5
- Transform: display rotation tag of 90 degrees
- Reason: deterministic nonzero rotation-metadata edge case

Represent the transform explicitly in the candidate data. Do not alter the
original row-0 candidate; it remains the unmodified control.

### 2. Deterministic lossless remux

Extend `scripts/build_reference_fixture.py` so an explicit rotation transform:

- Remuxes the MP4 bytes without decoding or re-encoding video.
- Preserves all input streams and packet timestamps where PyAV permits.
- Calls `set_display_rotation(90)` only on the video stream.
- Produces a payload that decodes with rotation 90.
- Leaves candidates without a transform byte-for-byte unchanged.
- Records both the original source payload SHA-256 and the derived fixture
  payload SHA-256 plus the transform in the manifest.
- Remains deterministic across two consecutive builds in the pinned local
  environment.

Keep one-row Parquet iteration and one scan per source shard. The duplicate
source coordinate must not cause a second shard scan.

### 3. Tests

Create a tiny locally encoded MP4 in the test using PyAV. Test that:

- The original fixture row has rotation 0.
- The derived fixture row has rotation 90.
- Encoded dimensions and decoded frame count are preserved.
- The original payload hash is unchanged.
- Two builds produce identical transformed payload and fixture hashes.
- No `ParquetFile.read()` or multi-row video batch is introduced.

Do not invoke YOLO, use the network, or access FineVideo in tests.

### 4. Ten-video workflow updates

Update `REFERENCE.md`, `scripts/stage_reference_kaggle.py`, and
`kaggle/reference/reference.py` from 9 to 10 expected videos. Keep expected
count validation strict. Update the candidate table and total duration; the
derived clip adds another 35 seconds.

Do not create or commit `reference/expected.jsonl` or
`reference/metadata.json`; Codex will rebuild, probe, upload a new private
dataset version, and regenerate them on Kaggle after review.

## Constraints

- Follow `AGENTS.md`.
- Do not change production decoding/inference/aggregation code, model settings,
  source selection other than the explicit derived row, Phase 2 evidence,
  `WRITEUP.md`, or `.env`.
- Do not begin batching, FP16, frame skipping, or other optimization.
- Do not claim the derived row is an original FineVideo rotation tag; document
  it clearly as a deterministic metadata transform.

## Verification

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-phase3-rotation
python -m compileall -q run.py prepare.py src scripts tests kaggle
python scripts/build_reference_fixture.py --shards-dir finevideo/data --output-dir artifacts/reference/input --selection reference/candidates.json --force
python scripts/probe_reference_fixture.py --fixture artifacts/reference/input/reference.parquet --output artifacts/reference/probe-local.json
```

The real probe must report 10 videos and `has_nonzero_rotation: true`, with the
derived row reporting rotation 90. Record the new fixture SHA-256.

Update `ai_chats/GEMINI_PHASE3_RESULT.md` with all changes and exact results,
then stop for Codex review.
