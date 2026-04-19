import time

import pycolmap


def test_timer_init():
    timer = pycolmap.Timer()
    assert timer is not None


def test_timer_start():
    timer = pycolmap.Timer()
    timer.start()


def test_timer_restart():
    timer = pycolmap.Timer()
    timer.start()
    timer.restart()


def test_timer_pause():
    timer = pycolmap.Timer()
    timer.start()
    timer.pause()


def test_timer_resume():
    timer = pycolmap.Timer()
    timer.start()
    timer.pause()
    timer.resume()


def test_timer_reset():
    timer = pycolmap.Timer()
    timer.start()
    timer.reset()


def test_timer_elapsed_seconds():
    timer = pycolmap.Timer()
    timer.start()
    time.sleep(0.001)
    elapsed = timer.elapsed_seconds()
    assert elapsed > 0.0


def test_timer_elapsed_micro_seconds():
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_micro_seconds()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_elapsed_minutes():
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_minutes()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_elapsed_hours():
    timer = pycolmap.Timer()
    timer.start()
    elapsed = timer.elapsed_hours()
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timer_print_seconds():
    timer = pycolmap.Timer()
    timer.start()
    timer.print_seconds()


def test_timer_print_minutes():
    timer = pycolmap.Timer()
    timer.start()
    timer.print_minutes()


def test_timer_print_hours():
    timer = pycolmap.Timer()
    timer.start()
    timer.print_hours()
