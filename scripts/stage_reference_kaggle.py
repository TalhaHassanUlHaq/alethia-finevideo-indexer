from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_reference_fixture import file_sha256
from scripts.stage_kaggle import add_path


OWNER = "talhahassanulhaq"
DATASET_SLUG = "althia-reference-suite"
SOURCE_PATHS = (
    "run.py",
    "prepare.py",
    "requirements.txt",
    "src",
    "scripts/__init__.py",
    "scripts/probe_reference_fixture.py",
    "scripts/generate_reference.py",
    "scripts/compare_reference.py",
    "scripts/validate_jsonl.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage the private Phase 3 reference fixture for Kaggle."
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("artifacts/reference/input"),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("artifacts/kaggle_reference_dataset"),
    )
    args = parser.parse_args()

    fixture_dir = args.fixture_dir
    if not fixture_dir.is_absolute():
        fixture_dir = ROOT / fixture_dir
    fixture = fixture_dir / "reference.parquet"
    manifest = fixture_dir / "reference-manifest.json"
    if not fixture.is_file() or not manifest.is_file():
        raise FileNotFoundError(
            "reference fixture is incomplete; build it before staging Kaggle"
        )

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    actual_hash = file_sha256(fixture)
    if actual_hash != manifest_data.get("fixture_sha256"):
        raise ValueError("reference fixture hash does not match its manifest")
    if manifest_data.get("video_count") != 10:
        raise ValueError("reference fixture must contain exactly 10 videos")

    stage = args.stage_dir
    if not stage.is_absolute():
        stage = ROOT / stage
    stage = stage.resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(fixture, stage / fixture.name)
    shutil.copy2(manifest, stage / manifest.name)

    with zipfile.ZipFile(
        stage / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in SOURCE_PATHS:
            add_path(archive, ROOT, relative)

    metadata = {
        "title": "Althia Phase 3 Reference Suite",
        "id": f"{OWNER}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
    }
    (stage / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
