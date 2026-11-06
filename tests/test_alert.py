from roomwatch.labels import LABELS
from roomwatch.session import Session
from roomwatch.alert import Alert


def standing_scores():
    scores = [0.0] * len(LABELS)
    scores[LABELS.index("standing")] = 0.9
    scores[LABELS.index("possible_fall")] = 0.1
    return scores


def fall_scores():
    scores = [0.0] * len(LABELS)
    scores[LABELS.index("possible_fall")] = 0.8
    scores[LABELS.index("lying")] = 0.2
    return scores


def test_red_then_ask_then_escalate():
    alert = Alert(red_hold_s=10.0, ask_s=20.0)
    assert alert.update(0.0, fall_scores()) == "red_hold"
    assert alert.update(9.9, fall_scores()) is None
    assert alert.update(10.0, fall_scores()) == "asking"
    assert alert.update(29.9, fall_scores()) is None
    assert alert.update(30.0, fall_scores()) == "escalate"
    assert alert.state == "escalate"


def test_standing_cancels_the_hold():
    alert = Alert()
    alert.update(0.0, fall_scores())
    assert alert.update(4.0, standing_scores()) == "cleared"
    assert alert.state == "watching"
    message = alert.message("lounge", standing_scores(), "cleared")
    assert message["room"] == "lounge"
    assert "possible_fall" in message["scores"]
    assert "image" not in message


def test_session_message_only_when_the_state_changes():
    session = Session("lounge", half_life_s=0.2)
    # Synthetic scores go through keypoints, so drive the alert with a session
    # only for the message shape. The timer itself is covered above.
    quiet = session.update(0.0, None)
    assert quiet["event"] is None
    assert quiet["message"] is None
    assert quiet["leader"] == "no_person"
