"""Red hold, spoken check, then escalate. Standing or walking cancels it."""

from __future__ import annotations

from hearth.labels import POSSIBLE_FALL, RECOVERED

RED_HOLD_S = 10.0
ASK_S = 20.0
# The fall bar has to lead. A tiny lead is not enough to start the hold.
FALL_SHARE = 0.40


class Alert:
    def __init__(self, red_hold_s: float = RED_HOLD_S, ask_s: float = ASK_S):
        self.red_hold_s = red_hold_s
        self.ask_s = ask_s
        self.state = "watching"
        self.since = 0.0

    def update(self, t: float, scores: list[float]) -> str | None:
        """Return the message name when the state changes, else None."""
        leader = max(range(len(scores)), key=scores.__getitem__)
        fall = leader == POSSIBLE_FALL and scores[POSSIBLE_FALL] >= FALL_SHARE
        recovered = leader in RECOVERED
        if self.state == "watching":
            if fall:
                self.state = "red_hold"
                self.since = t
                return "red_hold"
            return None
        if recovered:
            self.state = "watching"
            self.since = t
            return "cleared"
        if self.state == "red_hold" and t - self.since >= self.red_hold_s:
            self.state = "asking"
            self.since = t
            return "asking"
        if self.state == "asking" and t - self.since >= self.ask_s:
            self.state = "escalate"
            self.since = t
            return "escalate"
        return None

    def message(self, room: str, scores: list[float], event: str) -> dict:
        """The short message a robot or a staff tablet subscribes to. No image."""
        from hearth.labels import LABELS

        return {
            "room": room,
            "event": event,
            "state": self.state,
            "scores": {LABELS[i]: round(scores[i], 4) for i in range(len(LABELS))},
        }
