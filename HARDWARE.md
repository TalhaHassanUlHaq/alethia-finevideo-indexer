# Hardware

## Full-run environment

- Run location/provider: Kaggle, distributed across three independently
  operated accounts and 18 chunk kernels.
- Operating system: Kaggle Linux container, kernel `6.6.137+` (Ubuntu-based).
- GPU model: NVIDIA Tesla T4. All generated chunk kernels used Kaggle machine
  shape `NvidiaTeslaT4`; returned runtime manifests also identify Tesla T4.
- GPU VRAM: 15,360 MiB (16 GB GDDR6).
- NVIDIA driver and CUDA runtime: driver 580.178.04; CUDA 12.8 through the
  installed PyTorch `cu128` build.
- CPU model: Intel Xeon CPU at 2.20 GHz in the observed Kaggle environment.
- CPU cores / threads: 4 vCPUs per chunk kernel.
- System RAM: 30 GB per chunk kernel.
- Disk: Kaggle ephemeral workspace storage; its physical SSD/HDD device model
  was not exposed to the run.

## Production execution

- Input: the first 50 FineVideo Parquet shards,
  `train-00000-of-01357.parquet` through
  `train-00049-of-01357.parquet` (23.12 GiB).
- Configuration: D2 bounded prefetch, FP16, batch size 8.
- Command template:
  `python run.py --input <chunk-folder> --output <chunk>.jsonl --decoder-strategy D2 --precision fp16 --batch-size 8`.
- Source commit used by every chunk:
  `8a0385f2191acdceb6e83f6b838cef9dc3bf97ae`.
- Coordinator-observed distributed run window:
  - Earliest recorded chunk start: 2026-11-21 12:58:14 UTC.
  - Final returned chunk ingestion: 2026-11-22 06:19:41 UTC.
  - Elapsed wall-clock window: 62,487 seconds (17 hours, 21 minutes,
    27 seconds).
- Directly recorded active GPU time for the 11 manifests that retained real
  start/end timing: 72,582 seconds (20 hours, 9 minutes, 42 seconds). The
  remaining imported chunk manifests did not preserve their original runtime,
  so no fabricated single-GPU-equivalent total is reported.

## Final output

- Videos: 1,650.
- JSONL records: 461,757.
- Error records: 0.
- Validation: passed schema, canonical ordering, contiguous-seconds, complete
  50-shard ownership, and global duplicate checks.
- `results.jsonl` SHA-256:
  `820ba8b31664d336d629ad092567de996ba8a49a2fce0bb2f6032c0146ee2146`.
- `results.jsonl.gz` SHA-256:
  `1622595ab54be0e4029e348891ed718a7f673036e4292bb31e2bfcc1a2a53468`.
