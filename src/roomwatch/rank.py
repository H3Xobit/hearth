"""Smooth the raw scores so a held posture climbs and a sudden change trades off."""

from __future__ import annotations

import math

from roomwatch.labels import LABELS
from roomwatch.pose_math import measure, raw_scores


class Ranker:
    """One stream. Call update with seconds and COCO points (x, y, confidence)."""

    def __init__(self, half_life_s: float = 0.6):
        if half_life_s <= 0:
            raise ValueError("half_life_s must be positive")
        self.half_life_s = half_life_s
        self.scores = [0.0] * len(LABELS)
        self.scores[LABELS.index("no_person")] = 1.0
        self._last_t = None
        self._previous = None

    def update(self, t: float, points) -> list[float]:
        dt = 0.25 if self._last_t is None else max(0.0, t - self._last_t)
        now = measure(points, self._previous, dt)
        raw = raw_scores(now)
        alpha = 1.0 - math.exp(-math.log(2.0) * dt / self.half_life_s)
        self.scores = [
            (1.0 - alpha) * old + alpha * new for old, new in zip(self.scores, raw)
        ]
        self._last_t = t
        if now is not None:
            self._previous = now
        else:
            self._previous = None
        return list(self.scores)

    def leader(self) -> str:
        index = max(range(len(self.scores)), key=self.scores.__getitem__)
        return LABELS[index]

    def ranked(self) -> list[tuple[str, float]]:
        order = sorted(range(len(self.scores)), key=self.scores.__getitem__, reverse=True)
        return [(LABELS[i], self.scores[i]) for i in order]
