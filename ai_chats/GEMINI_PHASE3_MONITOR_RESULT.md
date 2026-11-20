# Gemini Phase 3 Monitoring Result Record

## Job Details

- **Kernel**: `talhahassanulhaq/althia-phase-3-reference`
- **URL**: <https://www.kaggle.com/code/talhahassanulhaq/althia-phase-3-reference>
- **Kernel Version**: 3
- **Dataset**: `talhahassanulhaq/althia-reference-suite`
- **Terminal Status**: `KernelWorkerStatus.COMPLETE`

## Downloaded Output Paths

All output artifacts were downloaded into `artifacts/kaggle_reference_output_v3/`:

- `expected.jsonl`: `artifacts/kaggle_reference_output_v3/expected.jsonl`
- `metadata.json`: `artifacts/kaggle_reference_output_v3/metadata.json`
- `probe.json`: `artifacts/kaggle_reference_output_v3/probe.json`
- `althia-phase-3-reference.log`: `artifacts/kaggle_reference_output_v3/althia-phase-3-reference.log`

## Validation Results

1. JSONL Validation:
   ```powershell
   python scripts/validate_jsonl.py artifacts/kaggle_reference_output_v3/expected.jsonl --expected-videos 10 --no-errors
   ```
   Output:
   ```json
   {"errors": 0, "lines": 316, "videos": 10}
   ```
   Exit code: `0`

2. Exact Self-Comparison:
   ```powershell
   python scripts/compare_reference.py artifacts/kaggle_reference_output_v3/expected.jsonl --reference artifacts/kaggle_reference_output_v3/expected.jsonl --exact --json
   ```
   Output:
   ```json
   {
     "exact_mode": true,
     "passed": true,
     "reference_file": "artifacts\\kaggle_reference_output_v3\\expected.jsonl",
     "candidate_file": "artifacts\\kaggle_reference_output_v3\\expected.jsonl",
     "total_expected_records": 316,
     "total_candidate_records": 316,
     "common_records": 316,
     "missing_records": 0,
     "unexpected_records": 0,
     "record_coverage": 1.0,
     "object_metrics": {
       "true_positives": 211,
       "false_positives": 0,
       "false_negatives": 0,
       "precision": 1.0,
       "recall": 1.0,
       "micro_f1": 1.0,
       "threshold": 0.95,
       "passed": true
     },
     "people_metrics": {
       "exact_matches": 316,
       "exact_rate": 1.0,
       "exact_threshold": 0.85,
       "exact_passed": true,
       "within_one_matches": 316,
       "within_one_rate": 1.0,
       "within_one_threshold": 0.95,
       "within_one_passed": true
     },
     "exact_record_match": {
       "matches": 316,
       "rate": 1.0
     },
     "error_records": {
       "reference_errors": 0,
       "candidate_errors": 0,
       "exact_error_match": true,
       "reference_error_counts": [],
       "candidate_error_counts": []
     },
     "schema_failures": {
       "reference_failures": 0,
       "candidate_failures": 0
     }
   }
   ```
   Exit code: `0`

## Hashes and Elapsed Time

- **Fixture SHA-256**:
  `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` (confirmed in `metadata.json`)
- **Model Weight SHA-256**:
  `9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f` (confirmed in `metadata.json`)
- **Result SHA-256**:
  `7c71fe76b1c48f17159d58575913528d223aac1a2bd42680f5df60b63575765d` (computed and verified against file bytes)
- **Kaggle Elapsed Time**:
  `170.65621505200005` seconds (`170.65551079200003` seconds in pipeline summary)

## Probe Results

From `artifacts/kaggle_reference_output_v3/probe.json`:

- **Video count**: 10
- **Has nonzero rotation**: `true`
- **Warning**: `null` (no acceptance warning)
- **Row 0 (control)**:
  - Frame count: 1034
  - Encoded / decoded dimensions: `[480, 360]`
  - Rotation: `0`
- **Row 9 (derived 90° transform)**:
  - Frame count: 1034
  - Encoded / decoded dimensions: `[480, 360]`
  - Rotation: `90`

## Provenance Collision Verification

From `artifacts/kaggle_reference_output_v3/metadata.json`:

1. Unique Source Coordinates (`hashes["source_payload_hashes"]`):
   - `train-00000-of-01357.parquet:5`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
   - Maps correctly to the original FineVideo source payload hash (no collision or overwrite by the derived row).

2. Per-Fixture-Row List (`fixture_rows`):
   - Row 0:
     - `fixture_row`: 0
     - `source_file`: `train-00000-of-01357.parquet`
     - `source_row`: 5
     - `source_payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `transform`: `null`
   - Row 9:
     - `fixture_row`: 9
     - `source_file`: `train-00000-of-01357.parquet`
     - `source_row`: 5
     - `source_payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `payload_sha256`: `481bfd6f858ff1df08485a042d98fbb67ff929dd16fc89c9a744a7dc370102fb`
     - `transform`: `{"degrees": 90, "type": "rotate"}`
   - Confirmed: Rows 0 and 9 share the identical original source hash, have distinct fixture payload hashes, and only row 9 carries the 90-degree transform.

## Constraints Compliance

- Production code in `src/`, `run.py`, and `prepare.py` was untouched.
- Reference code, candidate definitions, fixture bytes, model settings, and Kaggle identifiers were untouched.
- No files were promoted into `reference/`; files remain staged in `artifacts/kaggle_reference_output_v3/` for Codex final review.
- Kaggle API token was never exposed or written to disk.
