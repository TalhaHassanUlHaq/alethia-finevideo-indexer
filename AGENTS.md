# Althia Agent Instructions

These instructions apply to the entire repository.

## Roles

- Codex is the orchestrator, planner, supervisor, and final reviewer.
- Gemini in Antigravity is the implementation worker only.
- Gemini does not research, plan the project, supervise work, or perform final
  reviews. Codex supplies bounded implementation tasks and reviews all changes.
- The candidate is the sole author of `WRITEUP.md`. Agents must not draft,
  rewrite, review, or supply prose for that file.

## Current phase

The correctness-first baseline is implemented and its three-video Kaggle smoke
run has completed successfully. Do not begin optimization work unless a task
file explicitly asks for it.

Before working, read:

1. `AI_Engineering_TakeHome_Brief.pdf`
2. `README.md`
3. `ai_chats/GEMINI_CONTEXT.md`
4. The implementation task named in that context file, when one is active

## Non-negotiable assignment behavior

- Use YOLO11l with image size 640, confidence 0.4, class-aware NMS IoU 0.7,
  and default letterbox preprocessing.
- Supply decoded frames to inference as BGR and honor video rotation metadata.
- Target object order is exactly: `person`, `car`, `dog`, `laptop`,
  `cell phone`.
- An object is present in second `t` when it appears in at least half of that
  second's decoded frames.
- `people` is the median per-frame person count, with a halfway fractional
  median floored.
- A broken video must produce a per-video error record without discarding prior
  results or stopping the folder run.
- The required entry point remains
  `python run.py --input <folder> --output results.jsonl`.

## Working rules

- Keep generated data and model weights under ignored paths such as
  `artifacts/`; never commit credentials, `.env`, raw FineVideo shards, model
  weights, or Kaggle outputs.
- Preserve the existing directory structure and keep changes narrowly scoped.
- Do not weaken tests or validators to make failures disappear.
- Use the existing dependencies and architecture unless a task explicitly
  authorizes a change.
- Gemini must implement only the bounded change described by the active task
  file, record the files changed and verification performed, and stop. Any
  investigation needed to define the task belongs to Codex.
- When an approved implementation task launches a long-running Kaggle,
  training, or benchmark job, Gemini is responsible for monitoring it to a
  terminal state and recording the final status and output locations. Codex
  should not continuously poll while the job runs; Codex resumes for review
  after Gemini reports completion.
- Never expose or copy the Kaggle API token.

## Verification

Run tests with a workspace-local temporary directory:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Gemini must not accept review-only or research-only tasks. Codex reviews the
implementation and decides whether further changes are required.
