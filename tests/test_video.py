from fractions import Fraction

import numpy as np

from src.althia.video import frame_timestamp, rotate_bgr


class Rational:
    def __init__(self, numerator: int, denominator: int) -> None:
        self.numerator = numerator
        self.denominator = denominator


class Frame:
    def __init__(self, pts: int | None, time_base: Rational | None) -> None:
        self.pts = pts
        self.time_base = time_base


def test_frame_timestamp_prefers_pts_and_uses_rate_fallback() -> None:
    assert frame_timestamp(Frame(90, Rational(1, 30)), 7, Rational(25, 1)) == 3
    assert frame_timestamp(Frame(None, None), 7, Rational(25, 1)) == Fraction(7, 25)


def test_rotation_undoes_counterclockwise_display_matrix() -> None:
    image = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    rotated = rotate_bgr(image, 90)
    assert rotated.flags.c_contiguous
    assert rotated.tolist() == [[4, 1], [5, 2], [6, 3]]
    assert rotate_bgr(image, 360).tolist() == image.tolist()

