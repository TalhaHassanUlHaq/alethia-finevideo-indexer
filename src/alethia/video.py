from __future__ import annotations

import io
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Iterable, Iterator

import numpy as np

from .config import PipelineConfig


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


def decode_video(
    video_bytes: bytes,
    thread_type: str | None = None,
    thread_count: int | None = None,
) -> Iterator[DecodedFrame]:
    import av

    if not video_bytes:
        raise ValueError("empty video payload")
    with av.open(io.BytesIO(video_bytes), mode="r") as container:
        if not container.streams.video:
            raise ValueError("container has no video stream")
        stream = container.streams.video[0]
        if thread_type is not None and thread_count is not None and thread_count > 0:
            stream.codec_context.thread_type = thread_type
            stream.codec_context.thread_count = thread_count
        fallback_rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        for frame_index, frame in enumerate(container.decode(stream)):
            timestamp = frame_timestamp(frame, frame_index, fallback_rate)
            image = frame.to_ndarray(format="bgr24")
            yield DecodedFrame(
                image=rotate_bgr(image, getattr(frame, "rotation", 0)),
                timestamp=timestamp,
            )


def decode_video_prefetched(
    video_bytes: bytes,
    thread_type: str | None = None,
    thread_count: int | None = None,
    capacity: int = 64,
) -> Iterator[DecodedFrame]:
    sentinel = object()
    q: queue.Queue[object] = queue.Queue(maxsize=capacity)
    exc_box: list[BaseException] = []
    stop_event = threading.Event()

    def producer() -> None:
        try:
            for frame in decode_video(
                video_bytes, thread_type=thread_type, thread_count=thread_count
            ):
                if stop_event.is_set():
                    break
                while not stop_event.is_set():
                    try:
                        q.put(frame, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            exc_box.append(exc)
        finally:
            while True:
                try:
                    q.put(sentinel, timeout=0.05)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        break

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    try:
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item  # type: ignore[misc]
        if exc_box:
            raise exc_box[0]
    finally:
        stop_event.set()
        while t.is_alive():
            try:
                drain = q.get_nowait()
                if drain is sentinel:
                    break
            except queue.Empty:
                time.sleep(0.005)
        t.join(timeout=2.0)


def get_decoder_for_config(
    config: PipelineConfig,
) -> Callable[[bytes], Iterable[DecodedFrame]]:
    if config.decoder_strategy == "D0":
        return decode_video
    elif config.decoder_strategy == "D1":
        return lambda vb: decode_video(
            vb,
            thread_type=config.decode_thread_type,
            thread_count=config.decode_threads,
        )
    elif config.decoder_strategy == "D2":
        return lambda vb: decode_video_prefetched(
            vb,
            capacity=config.prefetch_capacity,
        )
    elif config.decoder_strategy == "D3":
        return lambda vb: decode_video_prefetched(
            vb,
            thread_type=config.decode_thread_type,
            thread_count=config.decode_threads,
            capacity=config.prefetch_capacity,
        )
    else:
        raise ValueError(f"unknown decoder strategy: {config.decoder_strategy}")
