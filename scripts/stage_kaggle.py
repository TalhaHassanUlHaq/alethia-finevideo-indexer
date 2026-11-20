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

from scripts.build_smoke_dataset import build_smoke_dataset


OWNER = "talhahassanulhaq"
DATASET_SLUG = "althia-baseline-smoke"
SOURCE_PATHS = (
    "run.py",
    "prepare.py",
    "requirements.txt",
    "src",
    "scripts/validate_jsonl.py",
)


def add_path(archive: zipfile.ZipFile, root: Path, relative: str) -> None:
    path = root / relative
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and "__pycache__" not in child.parts:
                archive.write(child, child.relative_to(root).as_posix())
    elif path.is_file():
        archive.write(path, path.relative_to(root).as_posix())
    else:
        raise FileNotFoundError(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the private Kaggle smoke dataset.")
    parser.add_argument(
        "--source-parquet",
        type=Path,
        default=Path("finevideo/data/train-00000-of-01357.parquet"),
    )
    parser.add_argument(
        "--stage-dir", type=Path, default=Path("artifacts/kaggle_dataset")
    )
    args = parser.parse_args()
    root = ROOT
    stage = (root / args.stage_dir).resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    source_parquet = args.source_parquet
    if not source_parquet.is_absolute():
        source_parquet = root / source_parquet
    build_smoke_dataset(source_parquet, stage)
    with zipfile.ZipFile(
        stage / "source.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in SOURCE_PATHS:
            add_path(archive, root, relative)
    metadata = {
        "title": "Althia Baseline Smoke",
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

