from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecoderStrategy = Literal["D0", "D1", "D2", "D3"]
DecodeThreadType = Literal["AUTO", "FRAME"]
Precision = Literal["fp32", "fp16"]


@dataclass(frozen=True)
class PipelineConfig:
    decoder_strategy: DecoderStrategy = "D2"
    decode_thread_type: DecodeThreadType = "AUTO"
    decode_threads: int = 2
    prefetch_capacity: int = 64
    precision: Precision = "fp16"
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.decoder_strategy not in ("D0", "D1", "D2", "D3"):
            raise ValueError(f"invalid decoder_strategy: {self.decoder_strategy!r}")
        if self.decode_thread_type not in ("AUTO", "FRAME"):
            raise ValueError(f"invalid decode_thread_type: {self.decode_thread_type!r}")
        if self.decode_threads < 1:
            raise ValueError(f"decode_threads must be >= 1, got {self.decode_threads}")
        if self.prefetch_capacity < 1:
            raise ValueError(f"prefetch_capacity must be >= 1, got {self.prefetch_capacity}")
        if self.precision not in ("fp32", "fp16"):
            raise ValueError(f"invalid precision: {self.precision!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
