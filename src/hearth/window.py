"""Two seconds of torso motion. A fall is the drop across the window, not one frame.

A 2 second window is the length that beat both 1 second and 3 seconds for an
ST-GCN fall model on UP-Fall inertial clips (Sensors, 2023, 23(4):2153).
"""

from __future__ import annotations


class MotionWindow:
    def __init__(self, seconds: float = 2.0):
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self.seconds = seconds
        self._rows: list[tuple[float, float, float]] = []

    def push(self, t: float, measure_now) -> dict:
        if measure_now is not None:
            self._rows.append((t, measure_now["upright"], measure_now["drop"]))
        self._rows = [row for row in self._rows if t - row[0] <= self.seconds]
        if not self._rows:
            return {"max_drop": 0.0, "upright_loss": 0.0, "upright": 0.0}
        uprights = [row[1] for row in self._rows]
        drops = [row[2] for row in self._rows]
        return {
            "max_drop": max(drops),
            "upright_loss": max(0.0, uprights[0] - uprights[-1]),
            "upright": uprights[-1],
        }
