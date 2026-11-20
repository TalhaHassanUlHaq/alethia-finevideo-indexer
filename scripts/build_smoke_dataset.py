from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ROWS = (4, 5, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_smoke_dataset(
    source: Path, output_dir: Path, rows: tuple[int, ...] = DEFAULT_ROWS
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    if not rows:
        raise ValueError("at least one row is required")
    if min(rows) < 0 or max(rows) >= parquet.metadata.num_rows:
        raise IndexError(f"rows {rows} outside 0..{parquet.metadata.num_rows - 1}")
    table = parquet.read().take(pa.array(rows, type=pa.int64()))
    output = output_dir / "smoke.parquet"
    pq.write_table(table, output, compression="snappy")
    metadata = table.column("json").to_pylist()
    manifest = {
        "source_file": source.name,
        "source_rows": list(rows),
        "smoke_rows": list(range(len(rows))),
        "duration_seconds": [item.get("duration_seconds") for item in metadata],
        "resolution": [item.get("resolution") for item in metadata],
        "sha256": sha256(output),
    }
    manifest_path = output_dir / "smoke-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the three-video smoke parquet.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows", nargs="+", type=int, default=list(DEFAULT_ROWS))
    args = parser.parse_args()
    output, manifest = build_smoke_dataset(
        args.source, args.output_dir, tuple(args.rows)
    )
    print(output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

