# Implementation Task: Phase 2 Acceptance Coverage

## Objective

Close the remaining automated acceptance-coverage gaps for the already
implemented correctness-first baseline. This task does not authorize pipeline
optimization or architectural changes.

## Required changes

1. Correct the output example in `CURRENT_PIPELINE.md` so it uses the required
   key `"video"`, not `"video_id"`.
2. Add focused automated tests proving all of the following:
   - Video identifiers are exactly `<Parquet basename>:<zero-based row>`.
   - A missing elapsed second is emitted with `objects: []` and `people: 0`.
   - The second containing the final decoded frame is emitted even when it is a
     partial second.
   - A video that fails partway retains its completed second records and emits
     one short error record.
   - Processing continues to a subsequent video after a damaged video.
   - The produced mixed result/error JSONL passes the project validator when
     errors are allowed.
3. Reuse small fakes and temporary Parquet files. Do not invoke YOLO, download
   weights, access the network, or read the full FineVideo shards in tests.
4. Do not change production code unless a new test exposes a confirmed defect.
   If a production change becomes necessary, keep it minimal and explain it in
   the result record.

## Constraints

- Follow `AGENTS.md`.
- Do not edit `WRITEUP.md`, `.env`, model settings, dependencies, Kaggle
  configuration, or existing smoke-run evidence.
- Do not add annotation or optimization functionality.
- Keep identifiers and JSON keys exactly as specified in the assignment brief.

## Verification

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-phase2
python -m compileall -q run.py prepare.py src scripts tests kaggle
python scripts/validate_jsonl.py artifacts/kaggle_output/results-smoke.jsonl --expected-videos 3 --no-errors
```

## Completion record

Write `ai_chats/GEMINI_PHASE2_RESULT.md` containing:

- Files changed.
- Tests added.
- Whether production code changed and why.
- Exact verification results.

Then stop and return control to Codex for review.
