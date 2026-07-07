import pytest

import pycolmap


def test_timestamp_from_seconds():
    # Timestamps are integer nanoseconds.
    assert pycolmap.timestamp_from_seconds(1.5) == 1_500_000_000
    # Rounds to the nearest nanosecond.
    assert pycolmap.timestamp_from_seconds(1e-9) == 1
    assert pycolmap.timestamp_from_seconds(0.4e-9) == 0


def test_seconds_from_timestamp():
    assert pycolmap.seconds_from_timestamp(1_500_000_000) == pytest.approx(1.5)
    assert pycolmap.seconds_from_timestamp(0) == 0.0


def test_timestamp_roundtrip():
    for seconds in [0.0, 0.001, 1.25, 10.0]:
        t = pycolmap.timestamp_from_seconds(seconds)
        assert pycolmap.seconds_from_timestamp(t) == pytest.approx(seconds)


def test_timestamp_diff_seconds():
    t0 = pycolmap.timestamp_from_seconds(1.0)
    t1 = pycolmap.timestamp_from_seconds(2.5)
    assert pycolmap.timestamp_diff_seconds(t1, t0) == pytest.approx(1.5)
    # The difference may be negative.
    assert pycolmap.timestamp_diff_seconds(t0, t1) == pytest.approx(-1.5)
