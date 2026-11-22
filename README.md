# Althia Video Indexer

FineVideo indexing pipeline using YOLO11l. The required compatibility command
is retained, while the selected production configuration is documented for the
full 50-shard run.

## Install

The grading target is Ubuntu 22.04 with Python 3.11 and an NVIDIA driver that
supports CUDA 12. No additional `apt-get` packages are required because PyAV's
wheel bundles FFmpeg.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt .
python prepare.py
```

`prepare.py` downloads and validates `yolo11l.pt`. Run it while internet access
is available. The timed indexing command does not require internet access.

## Run

Required evaluation command:

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl
```

By default, the CLI uses the validated production configuration: bounded D2
prefetching (`prefetch_capacity=64`), FP16 inference, and batch size 8. This
matches the configuration used to produce the final 50-shard results. Output is
written after each video so a later damaged video does not discard earlier results.

To run the original correctness-first baseline explicitly:

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl --decoder-strategy D0 --precision fp32 --batch-size 1
```

The explicit baseline decodes and evaluates every frame in FP32, one frame at a time.
See [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) before public submission.

## Validate

Validate a generated output file using the runtime environment:

```bash
python scripts/validate_jsonl.py results.jsonl
```

To run the optional development test suite, install its separately pinned
dependencies first:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Submission artifacts

The root-level `results.jsonl.gz` contains the merged output from all first 50
FineVideo shards: 461,757 records for 1,650 videos with zero error records. Its
SHA-256 is
`1622595ab54be0e4029e348891ed718a7f673036e4292bb31e2bfcc1a2a53468`.
The measured distributed Kaggle environment and run duration are recorded in
`HARDWARE.md`. The candidate is solely responsible for `WRITEUP.md` and for
adding the complete Codex and Gemini share links or exports under `ai_chats/`.

