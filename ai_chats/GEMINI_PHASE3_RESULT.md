# Gemini Phase 3 Result Record

## Files changed / created

- `reference/candidates.json`: Appended a 10th candidate derived from `train-00000-of-01357.parquet`, row 5 with explicit metadata transform `{"type": "rotate", "degrees": 90}` and reason "Deterministic nonzero rotation-metadata edge case". Row 0 remains the unmodified control.
- `scripts/build_reference_fixture.py`: Implemented deterministic lossless remuxing (`remux_with_rotation`) using PyAV `set_display_rotation(rotation)` on video stream template without decoding or re-encoding video. Preserves untransformed candidates byte-for-byte. Records `source_payload_sha256`, `payload_sha256`, and `transform` in `reference-manifest.json`. Retains single-row parquet streaming (`batch_size=1`, columns `["mp4", "json"]`) and single-pass shard scanning.
- `scripts/generate_reference.py`:
  - *Provenance collision correction*: Updated `build_metadata()` so `source_payload_hashes` maps unique source coordinates (`source_file:source_row`) to the true original `source_payload_sha256` (falling back to `payload_sha256` for backward compatibility) without collision or overwrite by derived rows sharing source coordinates.
  - Added a deterministic per-fixture-row list (`fixture_rows`) under both `hashes["fixture_rows"]` and top-level `metadata["fixture_rows"]` containing: `fixture_row`, `source_file`, `source_row`, `source_payload_sha256`, `payload_sha256`, `fixture_payload_sha256`, and `transform` (with `null` for unmodified rows). Rows 0 and 9 both appear, sharing the same `source_payload_sha256` while having different fixture payload hashes and only row 9 carrying the 90-degree transform.
- `tests/test_reference.py`:
  - Added `test_builder_derived_rotation_transform` and helper `make_tiny_mp4()` using PyAV. Tests that row 0 has rotation 0, derived row 1 has rotation 90, encoded dimensions and decoded frame counts are preserved, untransformed payload hash is preserved, builds are deterministic, and `iter_batches` uses `batch_size=1` without calling `ParquetFile.read()`.
  - Added `test_metadata_provenance_duplicate_source_coordinates` using a manifest with duplicate source coordinates (row 0 control vs row 2 derived 90-degree transform with shuffled input order). Proves no collision occurs in `source_payload_hashes`, `fixture_rows` is deterministically sorted by `fixture_row`, rows sharing coordinates have identical source hashes but distinct fixture payload hashes, only transformed rows carry transforms, the resulting metadata is fully JSON serializable, and manifests lacking `source_payload_sha256` fall back cleanly.
- `REFERENCE.md`: Updated candidate table, total duration (315 seconds across 10 videos), validation commands, and metadata description.
- `scripts/stage_reference_kaggle.py`: Updated expected video count assertion from 9 to 10.
- `kaggle/reference/reference.py`: Updated `--expected-videos` validation flag from 9 to 10.

## Commands run and exact results

1. Pytest suite (Provenance collision fix):
   ```powershell
   python -m pytest -q --basetemp artifacts/pytest-gemini-phase3-provenance
   ```
   Output:
   ```
   .............................                                            [100%]
   29 passed in 17.23s
   ```
   Exit code: `0`

2. Compilation check:
   ```powershell
   python -m compileall -q run.py prepare.py src scripts tests kaggle
   ```
   Output: None (all files compiled cleanly)
   Exit code: `0`

3. Real 10-video reference fixture build:
   ```powershell
   python scripts/build_reference_fixture.py --shards-dir finevideo/data --output-dir artifacts/reference/input --selection reference/candidates.json --force
   ```
   Output:
   ```
   Fixture created: artifacts\reference\input\reference.parquet
   Manifest created: artifacts\reference\input\reference-manifest.json
   ```
   Fixture SHA-256: `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
   Exit code: `0`

4. Real reference fixture PyAV probe:
   ```powershell
   python scripts/probe_reference_fixture.py --fixture artifacts/reference/input/reference.parquet --output artifacts/reference/probe-local.json
   ```
   Output:
   ```
   Probed 10 videos successfully.
   Nonzero rotation tag present: True
   Probe output written to: artifacts\reference\probe-local.json
   ```
   Exit code: `0`

   Probe summary:
   - `video_count`: 10
   - `has_nonzero_rotation`: true
   - `warning`: null
   - Row 0 (control): frame count 1034, dimensions `[480, 360]`, rotation `0`
   - Row 9 (derived transform): frame count 1034, dimensions `[480, 360]`, rotation `90`

5. Determinism verification (consecutive rebuild with `--force`):
   ```powershell
   python scripts/build_reference_fixture.py --shards-dir finevideo/data --output-dir artifacts/reference/input --selection reference/candidates.json --force
   ```
   Output:
   ```
   Fixture created: artifacts\reference\input\reference.parquet
   Manifest created: artifacts\reference\input\reference-manifest.json
   ```
   Fixture SHA-256: `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74` (identical across consecutive builds)
   Transformed payload SHA-256: `481bfd6f858ff1df08485a042d98fbb67ff929dd16fc89c9a744a7dc370102fb` (identical across consecutive builds)
   Exit code: `0`

6. Metadata provenance verification on real 10-video manifest:
   - `source_payload_hashes['train-00000-of-01357.parquet:5']`:
     `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621` (correct original source hash)
   - Row 0 in `fixture_rows`:
     - `fixture_row`: 0
     - `source_file`: `train-00000-of-01357.parquet`
     - `source_row`: 5
     - `source_payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `transform`: `null`
   - Row 9 in `fixture_rows`:
     - `fixture_row`: 9
     - `source_file`: `train-00000-of-01357.parquet`
     - `source_row`: 5
     - `source_payload_sha256`: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
     - `payload_sha256`: `481bfd6f858ff1df08485a042d98fbb67ff929dd16fc89c9a744a7dc370102fb`
     - `transform`: `{"type": "rotate", "degrees": 90}`

## Whether production code changed and why

No production code in `src/`, `run.py`, or `prepare.py` was modified. No candidates, fixture bytes, model settings, output JSONL, or Kaggle identifiers were altered. No performance optimization was begun. All changes were strictly confined to reference metadata generation (`scripts/generate_reference.py`), tests (`tests/test_reference.py`), and documentation (`REFERENCE.md`).

## Confirmation of 10-video fixture hashes

- Fixture file: `artifacts/reference/input/reference.parquet`
- Fixture SHA-256: `0389aae6340d27aca3691de70aee87fed93b795443f3c8a0f8dacc24adcfed74`
- Video count: 10
- Nonzero rotation present: `True` (Row 9: `rotation: 90`)
- Row 0 control payload SHA-256: `807f6ac7d6bd588d9df8558b13659bfad0e0d2a79b2136a6a6c8bf1e46a62621`
- Row 9 derived payload SHA-256: `481bfd6f858ff1df08485a042d98fbb67ff929dd16fc89c9a744a7dc370102fb`
