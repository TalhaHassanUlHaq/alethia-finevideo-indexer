# Worker Task: No-Run Release Preparation

## Scope

Perform only code/documentation/repository preparation. **Do not load YOLO,
run model inference, launch a Kaggle kernel, call the Kaggle API, download
FineVideo, or run any benchmark.** The user will transfer the repository to a
friend with a GPU for the real full-dataset run.

Read `AGENTS.md`, `CODEX.md`, and this task. Do not modify `WRITEUP.md`,
`.env`, immutable reference files, existing raw benchmark measurements, or
prior benchmark outputs. Do not commit.

## Required corrections

Implement the code-only parts of
`ai_chats/GEMINI_TASK_PHASE5_CORRECTIONS.md`, without its Kaggle/finalist-run
section:

1. Make `scripts/benchmark_configurations_driver.py` derive report claims from
   actual result rows. Remove hard-coded, contradictory performance/ranking
   statements. Where three-repeat finalist data is unavailable for a candidate,
   label it a screening result rather than a finalist conclusion.
2. Correct `summary.csv` speedup methodology in the code: finalist speedup is
   `baseline_finalist_median_runtime / candidate_finalist_median_runtime`; the
   baseline must be 1.000x. Do not fabricate a D1 finalist row—the friend may
   run it later.
3. Strengthen `scripts/compare_predictions.py` and focused tests so malformed
   schemas, duplicate `(video, second)` keys, unknown/canonically misordered
   objects, and differing error records cannot yield `exact_match=True`.
4. Create `RUN_ON_FRIEND.md`, a factual operational handoff for the GPU owner.
   It must include only commands/instructions, not fabricated measurements.
   Use this production command exactly:

   ```bash
   python run.py --input <folder-with-50-parquet-shards> --output results.jsonl --decoder-strategy D2 --precision fp16 --batch-size 16
   ```

   Cover environment setup, `python prepare.py`, the expected 50 input shards,
   model/GPU availability checks, output validation, gzip creation, recording
   hardware/start/end time facts for `HARDWARE.md`, and copying back only safe
   artifacts. State that the user must author `WRITEUP.md` themselves.
5. Update `README.md` only as needed to link this run guide and retain the
   original default command as the compatibility command.

## Verification allowed

Only CPU-safe checks are allowed:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Do not invoke `run.py`, detector/model constructors, benchmark workers, or
any test that requires a real model/GPU. If a test unexpectedly attempts this,
skip that test only with a documented reason; do not weaken it.

## Completion

Write `ai_chats/GEMINI_PHASE5_NO_RUN_RELEASE_PREP_RESULT.md` with files
changed, tests, a clear statement that no model/GPU/Kaggle execution occurred,
and remaining actions for the friend. Stop for Codex review.
