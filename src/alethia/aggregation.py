from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable

from .constants import TARGET_CLASS_IDS, TARGET_OBJECTS


@dataclass
class SecondBucket:
    frame_count: int = 0
    object_hits: Counter[str] = field(default_factory=Counter)
    people_counts: list[int] = field(default_factory=list)

    def add(self, detected_class_ids: Iterable[int]) -> None:
        class_ids = list(detected_class_ids)
        detected_names = {
            TARGET_CLASS_IDS[class_id]
            for class_id in class_ids
            if class_id in TARGET_CLASS_IDS
        }
        self.frame_count += 1
        self.object_hits.update(detected_names)
        self.people_counts.append(sum(class_id == 0 for class_id in class_ids))


def floored_median(values: list[int]) -> int:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


class VideoAggregation:
    def __init__(self) -> None:
        self._first_timestamp: Fraction | None = None
        self._buckets: dict[int, SecondBucket] = {}

    @property
    def frame_count(self) -> int:
        return sum(bucket.frame_count for bucket in self._buckets.values())

    def add_frame(
        self, timestamp: Fraction, detected_class_ids: Iterable[int]
    ) -> None:
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
        elapsed = timestamp - self._first_timestamp
        if elapsed < 0:
            elapsed = Fraction(0)
        second = elapsed.numerator // elapsed.denominator
        self._buckets.setdefault(second, SecondBucket()).add(detected_class_ids)

    def records(self, video_id: str) -> list[dict[str, object]]:
        if not self._buckets:
            return []
        records: list[dict[str, object]] = []
        for second in range(max(self._buckets) + 1):
            bucket = self._buckets.get(second)
            if bucket is None:
                objects: list[str] = []
                people = 0
            else:
                objects = [
                    name
                    for name in TARGET_OBJECTS
                    if bucket.object_hits[name] * 2 >= bucket.frame_count
                ]
                people = floored_median(bucket.people_counts)
            records.append(
                {
                    "video": video_id,
                    "second": second,
                    "objects": objects,
                    "people": people,
                }
            )
        return records

