import time

import pycolmap


def test_timer_init() -> None:
    timer = pycolmap.Timer()
    assert timer is not None


def test_timer_start() -> None:
    timer = pycolmap.Timer()
    timer.start()


def test_timer_restart() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.restart()


def test_timer_pause() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.pause()


def test_timer_resume() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.pause()
    timer.resume()


def test_timer_reset() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.reset()


def test_timer_elapsed_seconds() -> None:
    timer = pycolmap.Timer()
    timer.start()
    time.sleep(0.001)
    elapsed = timer.elapsed_seconds()
    assert elapsed > 0.0


def test_timer_elapsed_micro_seconds() -> None:
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_micro_seconds()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_elapsed_minutes() -> None:
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_minutes()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_elapsed_hours() -> None:
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_hours()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_print_seconds() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.print_seconds()


def test_timer_print_minutes() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.print_minutes()


def test_timer_print_hours() -> None:
    timer = pycolmap.Timer()
    timer.start()
    timer.print_hours()
