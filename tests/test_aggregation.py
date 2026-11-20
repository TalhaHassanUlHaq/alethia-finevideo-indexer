from fractions import Fraction

from src.althia.aggregation import VideoAggregation, floored_median


def test_floored_median_for_odd_even_and_fractional_cases() -> None:
    assert floored_median([3]) == 3
    assert floored_median([3, 1, 2]) == 2
    assert floored_median([0, 1]) == 0
    assert floored_median([2, 4]) == 3


def test_majority_people_and_missing_seconds() -> None:
    aggregation = VideoAggregation()
    aggregation.add_frame(Fraction(10, 1), [0, 16])
    aggregation.add_frame(Fraction(21, 2), [0, 0])
    aggregation.add_frame(Fraction(11, 1), [16])
    aggregation.add_frame(Fraction(23, 2), [])
    aggregation.add_frame(Fraction(13, 1), [2])

    assert aggregation.records("sample.parquet:0") == [
        {"video": "sample.parquet:0", "second": 0, "objects": ["person", "dog"], "people": 1},
        {"video": "sample.parquet:0", "second": 1, "objects": ["dog"], "people": 0},
        {"video": "sample.parquet:0", "second": 2, "objects": [], "people": 0},
        {"video": "sample.parquet:0", "second": 3, "objects": ["car"], "people": 0},
    ]


def test_missing_elapsed_second_emitted_with_empty_objects_and_zero_people() -> None:
    aggregation = VideoAggregation()
    aggregation.add_frame(Fraction(0, 1), [0])
    aggregation.add_frame(Fraction(25, 10), [2])

    records = aggregation.records("sample.parquet:0")
    assert records == [
        {"video": "sample.parquet:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "sample.parquet:0", "second": 1, "objects": [], "people": 0},
        {"video": "sample.parquet:0", "second": 2, "objects": ["car"], "people": 0},
    ]


def test_partial_second_holding_final_frame_is_emitted() -> None:
    aggregation = VideoAggregation()
    aggregation.add_frame(Fraction(0, 1), [0])
    aggregation.add_frame(Fraction(105, 100), [16])

    records = aggregation.records("sample.parquet:0")
    assert records == [
        {"video": "sample.parquet:0", "second": 0, "objects": ["person"], "people": 1},
        {"video": "sample.parquet:0", "second": 1, "objects": ["dog"], "people": 0},
    ]

