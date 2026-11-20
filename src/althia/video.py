from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class DecodedFrame:
    image: np.ndarray
    timestamp: Fraction


def _as_fraction(value: object) -> Fraction:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is None or denominator in (None, 0):
        raise ValueError(f"cannot convert {value!r} to a rational timestamp")
    return Fraction(int(numerator), int(denominator))


def frame_timestamp(frame: object, frame_index: int, fallback_rate: object) -> Fraction:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is not None and time_base is not None:
        return int(pts) * _as_fraction(time_base)
    rate = _as_fraction(fallback_rate)
    if rate <= 0:
        raise ValueError("video has no usable timestamps or frame rate")
    return Fraction(frame_index, 1) / rate


def rotate_bgr(image: np.ndarray, rotation: float | int | None) -> np.ndarray:
    if rotation is None:
        return np.ascontiguousarray(image)
    turns = round(float(rotation) / 90.0) % 4
    if turns:
        image = np.rot90(image, k=-turns, axes=(0, 1))
    return np.ascontiguousarray(image)


def decode_video(video_bytes: bytes) -> Iterator[DecodedFrame]:
    import av

    if not video_bytes:
        raise ValueError("empty video payload")
    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError("container has no video stream")
        stream = container.streams.video[0]
        fallback_rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = frame_timestamp(frame, frame_index, fallback_rate)
            image = frame.to_ndarray(format="bgr24")
            yield DecodedFrame(
                image=rotate_bgr(image, getattr(frame, "rotation", 0)),
                timestamp=timestamp,
            )

