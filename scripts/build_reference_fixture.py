from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SELECTION = ROOT / "reference" / "candidates.json"
DEFAULT_SHARDS_DIR = ROOT / "finevideo" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "reference" / "input"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def remux_with_rotation(video_bytes: bytes, rotation: int = 90) -> bytes:
    import av

    input_bio = io.BytesIO(video_bytes)
    output_bio = io.BytesIO()

    with av.open(input_bio, mode="r") as input_container:
        with av.open(output_bio, mode="w", format="mp4") as output_container:
            stream_map = {}
            for in_stream in input_container.streams:
                out_stream = output_container.add_stream_from_template(in_stream)
                if in_stream.type == "video":
                    out_stream.set_display_rotation(rotation)
                stream_map[in_stream] = out_stream

            for packet in input_container.demux():
                if packet.stream in stream_map and packet.dts is not None and packet.size > 0:
                    packet.stream = stream_map[packet.stream]
                    output_container.mux(packet)

    return output_bio.getvalue()


def load_candidates(selection_path: Path) -> list[dict[str, Any]]:
    if not selection_path.is_file():
        raise FileNotFoundError(f"Selection file not found: {selection_path}")
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(
            f"Selection file {selection_path} must be a non-empty list of candidate objects"
        )
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate index {index} is not a valid object")
        if "source_file" not in item or "source_row" not in item:
            raise ValueError(
                f"Candidate index {index} missing required 'source_file' or 'source_row'"
            )
    return data


def build_reference_fixture(
    shards_dir: Path,
    output_dir: Path,
    selection_path: Path = DEFAULT_SELECTION,
    force: bool = False,
) -> tuple[Path, Path]:
    candidates = load_candidates(selection_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_parquet = output_dir / "reference.parquet"
    manifest_path = output_dir / "reference-manifest.json"

    if fixture_parquet.exists() and not force:
        # Check if existing artifact is compatible
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_items = existing_manifest.get("items", [])
                candidate_keys = [
                    (c["source_file"], c["source_row"], str(c.get("transform"))) for c in candidates
                ]
                existing_keys = [
                    (item["source_file"], item["source_row"], str(item.get("transform"))) for item in existing_items
                ]
                current_sha = file_sha256(fixture_parquet)
                if (
                    candidate_keys == existing_keys
                    and current_sha == existing_manifest.get("fixture_sha256")
                ):
                    return fixture_parquet, manifest_path
            except Exception:
                pass
        raise FileExistsError(
            f"Fixture artifact already exists at {fixture_parquet}. Use --force to overwrite."
        )

    # Group candidate selections by source shard to scan each shard at most once
    shards_to_candidates: dict[str, list[int]] = {}
    for fixture_row, candidate in enumerate(candidates):
        shards_to_candidates.setdefault(candidate["source_file"], []).append(fixture_row)

    collected_tables: list[pa.Table | None] = [None] * len(candidates)
    items: list[dict[str, Any] | None] = [None] * len(candidates)

    for source_file, fixture_rows in shards_to_candidates.items():
        source_path = shards_dir / source_file
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Required source shard {source_file} not found in {shards_dir}"
            )

        parquet = pq.ParquetFile(source_path)
        schema_names = parquet.schema_arrow.names
        if "mp4" not in schema_names or "json" not in schema_names:
            raise ValueError(
                f"Source table {source_file} missing required columns 'mp4' and 'json'"
            )

        needed_source_rows: dict[int, list[int]] = {}
        for f_row in fixture_rows:
            s_row = int(candidates[f_row]["source_row"])
            if s_row < 0 or s_row >= parquet.metadata.num_rows:
                raise IndexError(
                    f"Row {s_row} outside range (0..{parquet.metadata.num_rows - 1}) in {source_file}"
                )
            needed_source_rows.setdefault(s_row, []).append(f_row)

        max_needed_row = max(needed_source_rows.keys())
        remaining_source_rows = set(needed_source_rows.keys())
        current_offset = 0

        # Bounded streaming batch iteration with single-row batches over only mp4 and json
        for batch in parquet.iter_batches(batch_size=1, columns=["mp4", "json"]):
            batch_len = len(batch)
            batch_end = current_offset + batch_len

            matched_rows = [
                r for r in remaining_source_rows if current_offset <= r < batch_end
            ]
            for r in matched_rows:
                local_idx = r - current_offset
                single_batch = batch.slice(local_idx, 1)
                row_table = pa.Table.from_batches([single_batch])

                mp4_bytes = single_batch.column(0)[0].as_py()
                json_meta = single_batch.column(1)[0].as_py()
                if json_meta is None:
                    json_meta = {}

                payload_digest = bytes_sha256(mp4_bytes)
                content_meta = json_meta.get("content_metadata")
                fps = (
                    content_meta.get("fps")
                    if isinstance(content_meta, dict)
                    else None
                ) or json_meta.get("fps")

                for f_row in needed_source_rows[r]:
                    cand = candidates[f_row]
                    duration = json_meta.get(
                        "duration_seconds", cand.get("duration_seconds")
                    )
                    resolution = json_meta.get("resolution")
                    reason = cand.get("reason") or cand.get("primary_coverage")

                    transform = cand.get("transform")
                    if transform:
                        if isinstance(transform, dict):
                            degrees = int(transform.get("degrees") or transform.get("rotation", 90))
                        elif isinstance(transform, (int, float)):
                            degrees = int(transform)
                        else:
                            degrees = 90
                        derived_mp4_bytes = remux_with_rotation(mp4_bytes, rotation=degrees)
                        derived_digest = bytes_sha256(derived_mp4_bytes)
                        derived_table = pa.Table.from_arrays(
                            [
                                pa.array([derived_mp4_bytes], type=pa.binary()),
                                pa.array([json_meta]),
                            ],
                            names=["mp4", "json"],
                        )
                        collected_tables[f_row] = derived_table
                        items[f_row] = {
                            "source_file": source_file,
                            "source_row": r,
                            "fixture_row": f_row,
                            "duration": duration,
                            "resolution": resolution,
                            "fps": fps,
                            "reason": reason,
                            "source_payload_sha256": payload_digest,
                            "payload_sha256": derived_digest,
                            "transform": transform,
                        }
                    else:
                        collected_tables[f_row] = row_table
                        items[f_row] = {
                            "source_file": source_file,
                            "source_row": r,
                            "fixture_row": f_row,
                            "duration": duration,
                            "resolution": resolution,
                            "fps": fps,
                            "reason": reason,
                            "source_payload_sha256": payload_digest,
                            "payload_sha256": payload_digest,
                            "transform": None,
                        }
                remaining_source_rows.remove(r)

            current_offset += batch_len
            if not remaining_source_rows or current_offset > max_needed_row:
                break

        if remaining_source_rows:
            raise RuntimeError(
                f"Failed to extract rows {remaining_source_rows} from {source_file}"
            )

    if any(t is None for t in collected_tables) or any(item is None for item in items):
        raise RuntimeError("Failed to collect all requested fixture rows")

    # Preserve exact frozen fixture order
    final_tables = [t for t in collected_tables if t is not None]
    final_items = [item for item in items if item is not None]

    combined_table = pa.concat_tables(final_tables)
    pq.write_table(combined_table, fixture_parquet, compression="snappy")

    fixture_digest = file_sha256(fixture_parquet)
    manifest = {
        "fixture_file": fixture_parquet.name,
        "fixture_sha256": fixture_digest,
        "video_count": len(final_items),
        "items": final_items,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return fixture_parquet, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic compact reference fixture from FineVideo shards."
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=DEFAULT_SHARDS_DIR,
        help="Directory containing FineVideo parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where fixture parquet and manifest will be written.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Path to candidate selection JSON file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing fixture artifacts.",
    )
    args = parser.parse_args()

    fixture, manifest = build_reference_fixture(
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        selection_path=args.selection,
        force=args.force,
    )
    print(f"Fixture created: {fixture}")
    print(f"Manifest created: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
