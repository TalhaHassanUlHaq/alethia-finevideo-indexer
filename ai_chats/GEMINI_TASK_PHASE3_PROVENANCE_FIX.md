# Implementation Task: Phase 3 Provenance Collision Fix

## Context

The final ten-video Kaggle run succeeded and is already validated:

- 316 JSONL records, 10 videos, zero errors
- Exact self-comparison passed
- Fixture SHA-256:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- Result SHA-256:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d`
- Rotation probe: row 0 rotation 0, derived row 9 rotation 90

Codex found one metadata-only defect. `build_metadata()` stores payload hashes
in a dictionary keyed by `<source_file>:<source_row>`. Rows 0 and 9 share the
same source coordinate, so row 9 overwrites row 0 and incorrectly labels the
derived payload hash as the source hash.

## Required correction

Update `scripts/generate_reference.py` metadata generation so it records,
without collisions:

1. Unique original source payload hashes keyed by source coordinate, using
   `source_payload_sha256` from the manifest (fall back to `payload_sha256` only
   for backward-compatible manifests).
2. A deterministic per-fixture-row list containing:
   - fixture row
   - source filename and source row
   - original source payload SHA-256
   - fixture payload SHA-256
   - transform, including `null` for unmodified rows

Rows 0 and 9 must both appear in the per-row list. They must have the same
original source hash, different fixture payload hashes, and only row 9 must
carry the 90-degree transform.

Add a focused unit test using a manifest with duplicate source coordinates that
proves no collision or mislabeling occurs and that the resulting metadata is
JSON serializable with deterministic row order.

## Constraints

- Follow `AGENTS.md`.
- Change only reference metadata code, tests, and related reference
  documentation if necessary.
- Do not change production code, candidates, fixture bytes, model settings,
  output JSONL, Kaggle identifiers, `WRITEUP.md`, or `.env`.
- Do not run inference or begin optimization.

## Verification

Run:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-phase3-provenance
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Update `ai_chats/GEMINI_PHASE3_RESULT.md` with the correction, test name, and
exact verification results. Then stop for Codex review.
