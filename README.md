# Althia Video Indexer

Correctness-first baseline for indexing FineVideo videos with YOLO11l.

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

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl
```

The baseline decodes and evaluates every frame in FP32, one frame at a time.
It writes output after each video so a later damaged video does not discard
earlier results.

## Validate

```bash
python scripts/validate_jsonl.py results.jsonl
pytest -q
```

## Kaggle smoke run

The tracked kernel is private and targets a T4 GPU. Stage the three-video fixture
and source snapshot, then create the private dataset and run the kernel:

```bash
python scripts/stage_kaggle.py
kaggle datasets create -p artifacts/kaggle_dataset -t
kaggle kernels push -p kaggle/baseline --accelerator NvidiaTeslaT4 --timeout 3600
kaggle kernels status talhahassanulhaq/althia-baseline
kaggle kernels logs talhahassanulhaq/althia-baseline
kaggle kernels output talhahassanulhaq/althia-baseline -p artifacts/kaggle_output
```

Set `KAGGLE_API_TOKEN` in the shell. Never commit credentials. The existing
legacy `.env` variable can be mapped into `KAGGLE_API_TOKEN` for the process.

### Verified baseline result

Private kernel version 3 completed successfully on Kaggle:

- Kernel: <https://www.kaggle.com/code/talhahassanulhaq/althia-baseline>
- Dataset: <https://www.kaggle.com/datasets/talhahassanulhaq/althia-baseline-smoke>
- Fixture: 3 videos, 104 seconds of metadata duration
- Output: 105 JSONL records, 3 videos, 0 error records
- Inference elapsed time: 74.454550787 seconds
- Hardware: 2 x Tesla T4 (the baseline intentionally used device 0 only)

The downloaded evidence is stored under ignored
`artifacts/kaggle_output/`. The extra output second is valid: the first video
contains a decoded frame in elapsed second 48.

## Assignment files

`HARDWARE.md`, `WRITEUP.md`, `results.jsonl.gz`, and complete `ai_chats/`
records are finalized after the optimized full 50-shard run. The candidate must
write `WRITEUP.md` without AI assistance.

