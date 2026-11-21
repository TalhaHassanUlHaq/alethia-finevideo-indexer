from fractions import Fraction
import queue
import threading
import time

import numpy as np
import pytest

from src.althia.config import PipelineConfig
from src.althia.pipeline import process_video
from src.althia.video import DecodedFrame, decode_video_prefetched, get_decoder_for_config


class FakeDetector:
    def detect_many(self, images: list[np.ndarray]) -> list[list[int]]:
        return [[0] if int(image[0, 0]) else [] for image in images]


class FailingDetector:
    def detect_many(self, images: list[np.ndarray]) -> list[list[int]]:
        raise RuntimeError("Inference crash on GPU")


def test_prefetched_decoder_ordering_and_final_partial_batch() -> None:
    def fake_source():
        for i in range(7):
            img = np.zeros((1, 1), dtype=np.uint8)
            img[0, 0] = i % 2
            yield DecodedFrame(image=img, timestamp=Fraction(i, 25))

    sentinel = object()
    q: queue.Queue[object] = queue.Queue(maxsize=3)

    def producer():
        for f in fake_source():
            q.put(f)
        q.put(sentinel)

    t = threading.Thread(target=producer)
    t.start()

    def fake_prefetched_decoder(_: bytes):
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item

    records, error = process_video(
        video_id="fake:0",
        video_bytes=b"dummy",
        detector=FakeDetector(),
        batch_size=4,
        decoder=fake_prefetched_decoder,
    )
    t.join()
    assert error is None
    assert len(records) == 1
    assert records[0]["video"] == "fake:0"


def test_prefetched_decoder_producer_exception_propagation() -> None:
    sentinel = object()
    q: queue.Queue[object] = queue.Queue(maxsize=4)
    exc_box = []

    def producer():
        try:
            for i in range(3):
                q.put(DecodedFrame(image=np.zeros((1, 1), dtype=np.uint8), timestamp=Fraction(i, 1)))
            raise ValueError("Corrupt packet in mid-stream")
        except Exception as exc:
            exc_box.append(exc)
        finally:
            q.put(sentinel)

    t = threading.Thread(target=producer)
    t.start()

    def failing_prefetched_decoder(_: bytes):
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        if exc_box:
            raise exc_box[0]

    records, error = process_video(
        video_id="fake:1",
        video_bytes=b"dummy",
        detector=FakeDetector(),
        batch_size=2,
        decoder=failing_prefetched_decoder,
    )
    t.join()
    assert "Corrupt packet in mid-stream" in str(error)
    # The first 2 frames flushed in batch 1, remaining partial frame flushed on exception
    assert len(records) == 3


def test_prefetched_decoder_consumer_crash_no_deadlock() -> None:
    # When consumer/detector crashes, producer shouldn't deadlock waiting to put into full queue
    sentinel = object()
    q: queue.Queue[object] = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    def infinite_producer():
        i = 0
        while not stop_event.is_set():
            frame = DecodedFrame(image=np.zeros((1, 1), dtype=np.uint8), timestamp=Fraction(i, 1))
            try:
                q.put(frame, timeout=0.05)
                i += 1
            except queue.Full:
                continue
        q.put(sentinel)

    t = threading.Thread(target=infinite_producer)
    t.start()

    def failing_consumer_decoder(_: bytes):
        try:
            for _ in range(5):
                yield q.get()
        finally:
            stop_event.set()
            while t.is_alive():
                try:
                    drain = q.get_nowait()
                    if drain is sentinel:
                        break
                except queue.Empty:
                    time.sleep(0.005)
            t.join(timeout=1.0)

    records, error = process_video(
        video_id="fake:2",
        video_bytes=b"dummy",
        detector=FailingDetector(),
        batch_size=2,
        decoder=failing_consumer_decoder,
    )
    assert not t.is_alive(), "Producer thread failed to join (deadlock occurred)"
    assert "Inference crash on GPU" in str(error)


def test_get_decoder_for_config() -> None:
    c0 = PipelineConfig(decoder_strategy="D0")
    c1 = PipelineConfig(decoder_strategy="D1", decode_threads=4, decode_thread_type="FRAME")
    c2 = PipelineConfig(decoder_strategy="D2", prefetch_capacity=32)
    c3 = PipelineConfig(decoder_strategy="D3", decode_threads=2, prefetch_capacity=16)

    d0 = get_decoder_for_config(c0)
    d1 = get_decoder_for_config(c1)
    d2 = get_decoder_for_config(c2)
    d3 = get_decoder_for_config(c3)

    assert callable(d0)
    assert callable(d1)
    assert callable(d2)
    assert callable(d3)
