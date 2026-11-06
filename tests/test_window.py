from roomwatch.window import MotionWindow


def test_window_forgets_a_drop_after_two_seconds():
    window = MotionWindow(2.0)
    window.push(0.0, {"upright": 1.0, "drop": 0.0})
    kept = window.push(0.4, {"upright": 0.1, "drop": 2.5})
    assert kept["max_drop"] == 2.5
    assert kept["upright_loss"] > 0.8
    later = window.push(2.5, {"upright": 0.05, "drop": 0.0})
    assert later["max_drop"] == 0.0


def test_empty_window_is_still():
    assert MotionWindow().push(1.0, None)["max_drop"] == 0.0
