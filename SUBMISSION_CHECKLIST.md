# Submission Checklist

This checklist follows Section 7 of `AI_Engineering_TakeHome_Brief.pdf`.
Complete every pending item only with evidence from the real full 50-shard run.

## Required root deliverables

- [x] `run.py` — required entry point; default invocation remains supported.
- [x] `requirements.txt` — pinned runtime dependencies.
- [x] `README.md` — installation, preparation, default run, optimized run, and
  validation instructions.
- [ ] `results.jsonl.gz` — create from the actual 50-shard output and commit at
  the repository root. It is intentionally absent until that run is complete.
- [ ] `HARDWARE.md` — replace every pending field with actual GPU-host facts
  and measured full-run duration.
- [ ] `WRITEUP.md` — author independently, without AI assistance.
- [ ] `ai_chats/` — add complete exported/shareable Codex and Gemini histories
  to `ai_chats/links.txt` or as native exports; remove only secrets.

## Before transferring to the GPU host

- [ ] Read `RUN_ON_FRIEND.md`.
- [ ] Ensure the host has Python 3.11, an NVIDIA GPU, current CUDA-capable
  driver, sufficient disk space, and the 50 Parquet shards.
- [ ] Run `python prepare.py` with internet access before the timed run.
- [ ] Verify `nvidia-smi` and `torch.cuda.is_available()`.

## Full-run command

```bash
python run.py --input /path/to/finevideo/data --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 8
```

## After the full run

- [ ] Validate: `python scripts/validate_jsonl.py results.jsonl`.
- [ ] Compress: `gzip -k results.jsonl`.
- [ ] Move `results.jsonl.gz` to this repository root and commit it. The
  `.gitignore` allows exactly this root deliverable.
- [ ] Record start/end time, hardware, output checksum, and validation result
  in `HARDWARE.md`.
- [ ] Review `git status --ignored` for raw shards, model weights, `.env`,
  logs, and other secrets before publishing.
- [ ] Confirm the last commit is made before the assignment deadline.
