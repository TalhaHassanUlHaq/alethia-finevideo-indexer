import pytest

from src.althia.config import PipelineConfig


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig()
    assert config.decoder_strategy == "D2"
    assert config.decode_thread_type == "AUTO"
    assert config.decode_threads == 2
    assert config.prefetch_capacity == 64
    assert config.precision == "fp16"
    assert config.batch_size == 8


def test_pipeline_config_explicit_baseline() -> None:
    config = PipelineConfig(
        decoder_strategy="D0",
        precision="fp32",
        batch_size=1,
    )
    assert config.decoder_strategy == "D0"
    assert config.precision == "fp32"
    assert config.batch_size == 1


def test_pipeline_config_validations() -> None:
    with pytest.raises(ValueError, match="invalid decoder_strategy"):
        PipelineConfig(decoder_strategy="INVALID")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid decode_thread_type"):
        PipelineConfig(decode_thread_type="SLICE")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="decode_threads must be >= 1"):
        PipelineConfig(decode_threads=0)

    with pytest.raises(ValueError, match="prefetch_capacity must be >= 1"):
        PipelineConfig(prefetch_capacity=0)

    with pytest.raises(ValueError, match="invalid precision"):
        PipelineConfig(precision="int8")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        PipelineConfig(batch_size=0)
