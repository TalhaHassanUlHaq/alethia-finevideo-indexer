# Worker Task: Final Submission CLI Defaults

## Objective

Make the assignment-required command use the validated full-run production
configuration by default, without changing model thresholds, preprocessing,
aggregation, output schema, or explicit baseline availability.

The evaluator runs exactly:

```bash
python run.py --input <folder> --output results.jsonl
```

The completed 50-shard results used D2 bounded prefetch, FP16, and batch size
8. The required unflagged command must therefore select that configuration.

## Required changes

1. Change CLI defaults only:
   - decoder strategy: `D2`
   - precision: `fp16`
   - batch size: `8`
   - retain prefetch capacity `64`
2. Keep the correctness-first baseline available explicitly:

   ```bash
   python run.py --input <folder> --output results.jsonl --decoder-strategy D0 --precision fp32 --batch-size 1
   ```

3. Update focused CLI/config tests so they verify both the optimized defaults
   and explicit baseline flags. Do not weaken any existing semantic tests.
4. Update `README.md` so the required command is accurately described as the
   optimized production configuration and the explicit baseline command is
   documented separately.
5. Do not modify `WRITEUP.md`, `results.jsonl.gz`, `HARDWARE.md`, reference
   artifacts, thresholds, aggregation, model weights, or generated run data.
6. Do not run inference, load YOLO weights, launch Kaggle, or access `.env`.

## Verification

Run CPU-safe checks only:

```powershell
python -m pytest -q --basetemp artifacts/pytest-gemini-final-defaults
python -m compileall -q run.py prepare.py src scripts tests kaggle
```

Write `ai_chats/GEMINI_FINAL_SUBMISSION_DEFAULTS_RESULT.md` with files changed,
test results, and an explicit statement that no inference/Kaggle/credentials
were used. Stop for Codex review. Do not commit.
