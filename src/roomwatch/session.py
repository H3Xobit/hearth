"""One room camera: smooth the bars, then run the alert timer."""

from __future__ import annotations

from roomwatch.alert import Alert
from roomwatch.rank import Ranker


class Session:
    def __init__(self, room: str, half_life_s: float = 0.6):
        self.room = room
        self.ranker = Ranker(half_life_s=half_life_s)
        self.alert = Alert()

    def update(self, t: float, points) -> dict:
        scores = self.ranker.update(t, points)
        event = self.alert.update(t, scores)
        message = None
        if event is not None:
            message = self.alert.message(self.room, scores, event)
        return {
            "scores": scores,
            "ranked": self.ranker.ranked(),
            "leader": self.ranker.leader(),
            "state": self.alert.state,
            "event": event,
            "message": message,
            "motion": dict(self.ranker.motion),
        }
