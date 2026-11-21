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
pip install -r requirements.txt
python prepare.py
```

`prepare.py` downloads and validates `yolo11l.pt`. Run it while internet access
is available. The timed indexing command does not require internet access.

## Run

Required compatibility command:

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl
```

The baseline decodes and evaluates every frame in FP32, one frame at a time.
It writes output after each video so a later damaged video does not discard
earlier results.

For the full 50-shard production run on a GPU host, use:

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 8
```

See [RUN_ON_FRIEND.md](RUN_ON_FRIEND.md) for the complete host handoff,
validation, compression, and hardware-recording steps. See
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) before public submission.

## Validate

```bash
python scripts/validate_jsonl.py results.jsonl
pytest -q
```

## Submission artifacts

The assignment requires root-level `results.jsonl.gz`, `HARDWARE.md`,
`WRITEUP.md`, and complete `ai_chats/` records. The compressed full-run output
and measured hardware report are pending the real GPU-host run. The candidate
must write `WRITEUP.md` independently, without AI assistance.

