from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import av
import pyarrow.parquet as pq


def portrait_rows(path: Path) -> set[int]:
    table = pq.read_table(path, columns=["json"])
    rows: set[int] = set()
    for index, metadata in enumerate(table.column("json").to_pylist()):
        resolution = (metadata or {}).get("resolution")
        if not isinstance(resolution, str) or "x" not in resolution:
            continue
        try:
            width, height = (int(value) for value in resolution.split("x", 1))
        except ValueError:
            continue
        if width < height:
            rows.add(index)
    return rows


def rotation_for_payload(payload: bytes) -> tuple[int, int, int] | None:
    with av.open(io.BytesIO(payload), mode="r") as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        frame = next(container.decode(stream), None)
        if frame is None:
            return None
        rotation = int(getattr(frame, "rotation", 0) or 0)
        if rotation == 0 and "rotate" in stream.metadata:
            rotation = int(stream.metadata["rotate"])
        return rotation, int(stream.width), int(stream.height)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find a short FineVideo row with a genuine nonzero rotation tag."
    )
    parser.add_argument("--shards-dir", type=Path, default=Path("finevideo/data"))
    parser.add_argument("--max-duration", type=int, default=90)
    args = parser.parse_args()

    checked = 0
    failures = 0
    for shard in sorted(args.shards_dir.glob("*.parquet")):
        candidates = portrait_rows(shard)
        if not candidates:
            continue
        parquet = pq.ParquetFile(shard)
        metadata = pq.read_table(shard, columns=["json"]).column("json").to_pylist()
        for row, batch in enumerate(
            parquet.iter_batches(batch_size=1, columns=["mp4"])
        ):
            if row not in candidates:
                continue
            duration = (metadata[row] or {}).get("duration_seconds")
            if not isinstance(duration, int) or duration > args.max_duration:
                continue
            checked += 1
            try:
                result = rotation_for_payload(batch.column(0)[0].as_py())
            except Exception:
                failures += 1
                continue
            if result is None:
                continue
            rotation, encoded_width, encoded_height = result
            if rotation % 360:
                print(
                    json.dumps(
                        {
                            "source_file": shard.name,
                            "source_row": row,
                            "duration_seconds": duration,
                            "resolution": (metadata[row] or {}).get("resolution"),
                            "encoded_dimensions": [encoded_width, encoded_height],
                            "rotation": rotation,
                            "checked": checked,
                            "failures": failures,
                        },
                        sort_keys=True,
                    )
                )
                return 0
    print(json.dumps({"found": False, "checked": checked, "failures": failures}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
