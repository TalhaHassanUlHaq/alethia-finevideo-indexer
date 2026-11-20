# Worker Task: Monitor Final Phase 3 Kaggle Run

## Job

- Private kernel: `talhahassanulhaq/althia-phase-3-reference`
- URL: <https://www.kaggle.com/code/talhahassanulhaq/althia-phase-3-reference>
- Expected active version: 3
- Dataset: `talhahassanulhaq/althia-reference-suite`
- Expected fixture: 10 videos

Codex has already launched the job. Do not push another kernel version.

## Required work

1. Monitor the kernel until Kaggle reports `COMPLETE`, `ERROR`, or another
   terminal state. Poll at a reasonable interval; do not modify code while it
   runs.
2. If complete, download outputs to
   `artifacts/kaggle_reference_output_v3/`. The Kaggle CLI may time out while
   downloading the copied model directory; resume with overwrite enabled until
   `expected.jsonl`, `metadata.json`, and `probe.json` are present.
3. Run:

```powershell
python scripts/validate_jsonl.py artifacts/kaggle_reference_output_v3/expected.jsonl --expected-videos 10 --no-errors
python scripts/compare_reference.py artifacts/kaggle_reference_output_v3/expected.jsonl --reference artifacts/kaggle_reference_output_v3/expected.jsonl --exact --json
```

4. Verify and record:
   - Fixture hash is
     `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`.
   - Probe reports 10 videos, no warning, and a nonzero rotation for fixture
     row 9.
   - Summary reports 10 videos and zero errors.
   - `source_payload_hashes["train-00000-of-01357.parquet:5"]` equals the
     original hash
     `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`.
   - Metadata contains fixture rows 0 and 9 with the same original source hash,
     different fixture payload hashes, and only row 9 has the 90-degree
     transform.
   - Record the result SHA-256 and elapsed time rather than assuming they match
     version 2.
5. Update `ai_chats/GEMINI_PHASE3_MONITOR_RESULT.md` with terminal status,
   validation outputs, hashes, elapsed time, probe result, provenance checks,
   and downloaded output paths.

If the kernel fails, record the terminal status and available error/log text;
do not edit code or relaunch it. Then notify Codex that monitoring is complete.

## Constraints

- Follow `AGENTS.md`.
- Never print, copy, or record the Kaggle token. Map it from `.env` only in the
  process environment.
- Do not change production code, reference code, candidates, expected outputs,
  `WRITEUP.md`, or `.env`.
- Do not promote files into `reference/`; Codex will do final review and
  promotion.
