# Hardware

> Status: pending the real 50-shard production run. Do not complete this file
> from estimates or benchmark fixtures; record the facts from the GPU host.

## Full-run environment

- Run location/provider:
- Operating system:
- GPU model:
- GPU VRAM:
- NVIDIA driver and CUDA runtime:
- CPU model:
- CPU cores / threads:
- System RAM:
- Disk type (SSD or HDD):

## Production execution

- Input: first 50 FineVideo Parquet shards
- Command:
  `python run.py --input <folder> --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 8`
- Start time (UTC):
- End time (UTC):
- Total wall-clock duration:
- Output SHA-256 (`results.jsonl.gz`):
- JSONL validator result:
- Notes on per-video error records, if any:

The assignment requires this report to describe the actual machine and complete
50-shard run that produced `results.jsonl.gz`.

