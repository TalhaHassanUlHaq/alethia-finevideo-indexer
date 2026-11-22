from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .constants import TARGET_OBJECTS


def validate_jsonl(
    path: Path, expected_videos: int | None = None, allow_errors: bool = True
) -> dict[str, int]:
    seconds: dict[str, set[int]] = defaultdict(set)
    videos: set[str] = set()
    errors = 0
    lines = 0
    canonical_index = {name: index for index, name in enumerate(TARGET_OBJECTS)}

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            video = record.get("video")
            if not isinstance(video, str) or not video:
                raise ValueError(f"line {line_number}: invalid video id")
            videos.add(video)
            if "error" in record:
                if set(record) != {"video", "error"} or not isinstance(
                    record["error"], str
                ):
                    raise ValueError(f"line {line_number}: invalid error record")
                errors += 1
                continue
            if set(record) != {"video", "second", "objects", "people"}:
                raise ValueError(f"line {line_number}: invalid result keys")
            second = record["second"]
            people = record["people"]
            objects = record["objects"]
            if not isinstance(second, int) or isinstance(second, bool) or second < 0:
                raise ValueError(f"line {line_number}: invalid second")
            if second in seconds[video]:
                raise ValueError(f"line {line_number}: duplicate second")
            seconds[video].add(second)
            if not isinstance(people, int) or isinstance(people, bool) or people < 0:
                raise ValueError(f"line {line_number}: invalid people count")
            if not isinstance(objects, list) or any(
                not isinstance(name, str) or name not in canonical_index
                for name in objects
            ):
                raise ValueError(f"line {line_number}: invalid objects")
            if len(set(objects)) != len(objects):
                raise ValueError(f"line {line_number}: duplicate object")
            if objects != sorted(objects, key=canonical_index.__getitem__):
                raise ValueError(f"line {line_number}: non-canonical object order")

    for video, observed in seconds.items():
        if observed and observed != set(range(max(observed) + 1)):
            raise ValueError(f"{video}: seconds are not contiguous from zero")
    if expected_videos is not None and len(videos) != expected_videos:
        raise ValueError(f"expected {expected_videos} videos, found {len(videos)}")
    if errors and not allow_errors:
        raise ValueError(f"found {errors} error records")
    return {"lines": lines, "videos": len(videos), "errors": errors}

