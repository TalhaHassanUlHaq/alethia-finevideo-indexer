# Current Pipeline

The current implementation is a correctness-first, single-GPU baseline for
indexing FineVideo videos with YOLO11l.

## Pipeline overview

```text
Folder of Parquet shards
        |
        v
Discover and sort *.parquet files
        |
        v
Read each row containing an encoded video
        |
        v
Decode every frame with PyAV
        |
        v
Apply rotation metadata and convert to BGR
        |
        v
Run YOLO11l in FP32 with batch size 1
        |
        v
Extract target detections and person counts
        |
        v
Group frames by elapsed second
        |
        v
Apply the assignment aggregation rules
        |
        v
Append validated records to results.jsonl
```

## 1. Entry point and input discovery

The required command is:

```bash
python run.py --input <folder> --output results.jsonl
```

The pipeline discovers Parquet files in the input folder and processes them in
a deterministic order. Each relevant Parquet row contains one encoded video.
The encoded bytes are passed to PyAV through an in-memory stream.

## 2. Video decoding

PyAV decodes every frame of each video. For each frame, the pipeline:

- Calculates elapsed time using `PTS * time_base`.
- Falls back to frame index and frame rate if a timestamp is unavailable.
- Assigns the frame to `floor(elapsed_seconds)`.
- Applies the frame's rotation metadata.
- Converts the decoded image to BGR.

The baseline does not skip frames.

Implementation: `src/althia/video.py`

## 3. YOLO inference

Each frame is currently passed to YOLO11l individually.

| Setting | Current value |
|---|---:|
| Model | `yolo11l.pt` |
| Image size | 640 |
| Confidence threshold | 0.4 |
| NMS IoU threshold | 0.7 |
| Precision | FP32 |
| Batch size | 1 |
| Device | GPU 0 |
| Preprocessing | Ultralytics default letterbox |

YOLO performs class-aware NMS over all classes. After inference, the pipeline
retains the five assignment targets:

| Target | COCO class ID |
|---|---:|
| `person` | 0 |
| `car` | 2 |
| `dog` | 16 |
| `laptop` | 63 |
| `cell phone` | 67 |

For every frame, the pipeline records:

- Which target classes appeared.
- The number of detected people.

Implementation: `src/althia/inference.py`

## 4. Per-second aggregation

Frames are grouped by elapsed second. For each target object, the object is
present when it appears in at least half of that second's decoded frames:

```text
frames containing object >= ceil(total frames / 2)
```

The `people` value is calculated by:

1. Collecting the person count from every frame in the second.
2. Calculating the median count.
3. Flooring the result when the median is fractional.

Objects are always emitted in this canonical order:

```text
person, car, dog, laptop, cell phone
```

Implementation: `src/althia/aggregation.py`

## 5. Output and failure handling

The pipeline writes one JSON object per elapsed second. A typical record is:

```json
{
  "video": "shard.parquet:12",
  "second": 4,
  "objects": ["person", "car"],
  "people": 2
}
```

Results are appended after each video. If a later video fails, records already
written for earlier videos remain available. A broken video produces a
per-video error record, and folder processing continues.

Implementation: `src/althia/pipeline.py`

## 6. Validation

The JSONL validator checks:

- Every line contains valid JSON.
- Required fields have valid types.
- Target objects use canonical ordering.
- A video does not contain duplicate seconds.
- Seconds are contiguous for each successfully processed video.
- The expected number of videos is present when specified.
- Error records are absent when `--no-errors` is requested.

Validation command:

```bash
python scripts/validate_jsonl.py results.jsonl
```

Implementation: `scripts/validate_jsonl.py`

## 7. Verified smoke-run result

The baseline completed successfully on a private three-video Kaggle fixture:

| Measurement | Result |
|---|---:|
| Videos | 3 |
| Metadata duration | 104 seconds |
| JSONL records | 105 |
| Error records | 0 |
| Inference elapsed time | 74.454550787 seconds |
| Local tests | 9 passed |

The 105 records are expected. One video contains a decoded frame in elapsed
second 48, and the assignment requires including the second that contains the
final frame.

## 8. Current limitation and next direction

The current execution pattern is serial:

```text
decode one frame -> infer one frame -> repeat
```

This prioritizes correctness but underutilizes the GPU. The next version should
preserve the same timestamps, aggregation semantics, output schema, and error
handling while introducing measured improvements such as:

- Batched inference.
- FP16 inference where output equivalence is acceptable.
- Better overlap between video decoding and GPU inference.
- Improved GPU utilization.
- Multi-GPU execution where the target environment supports it.

Every optimization must be compared with the correctness-first baseline before
it is accepted.

## Phase 2 acceptance status

Phase 2 is complete. The independently reviewed test suite contains 13 passing
tests and covers exact video identifiers, empty seconds, partial final seconds,
partway failures, preservation of completed records, continuation to subsequent
videos, JSONL validation, timestamp handling, rotation, aggregation, and error
formatting. The three-video Kaggle smoke output also passes validation with zero
error records.
