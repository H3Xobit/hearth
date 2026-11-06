"""Turn one body's COCO keypoints into an unsmoothed eight-way score.

Image y grows downward. A joint is (x, y, confidence). Missing joints are
confidence 0. The score is geometric. It is not a trained fall network.
"""

from __future__ import annotations

import math

from hearth.labels import LABELS

# COCO body order used by RTMPose when the openpose flag is off.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

CONF_MIN = 0.3
_LOGIT = {
    "standing": 0.4,
    "walking": -0.2,
    "sitting": -0.4,
    "using_an_object": -1.2,
    "unclear": -0.6,
    "no_person": -4.0,
    "lying": -0.8,
    "possible_fall": -1.0,
}


def _joint(points, index):
    if points is None or index >= len(points):
        return None
    x, y, conf = points[index]
    if conf < CONF_MIN:
        return None
    return (float(x), float(y), float(conf))


def _mid(a, b):
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def body_seen(points) -> bool:
    seen = 0
    for index in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        if _joint(points, index) is not None:
            seen += 1
    return seen >= 2


def measure(points, previous, dt):
    """Return uprightness, drop in torsos per second, step, bend, reach.

    previous is an earlier measure dict or None. dt is seconds.
    """
    if not body_seen(points):
        return None
    shoulder = _mid(_joint(points, L_SHOULDER), _joint(points, R_SHOULDER))
    hip = _mid(_joint(points, L_HIP), _joint(points, R_HIP))
    if shoulder is None or hip is None:
        return None
    torso = max(_dist(shoulder, hip), 1.0)
    # Shoulder above the hip in the image is the upright direction.
    # Horizontal offset is already inside torso, so upright is the vertical share.
    dy = shoulder[1] - hip[1]
    upright = max(-1.0, min(1.0, -dy / torso))
    bends = []
    for hip_i, knee_i, ankle_i in (
        (L_HIP, L_KNEE, L_ANKLE),
        (R_HIP, R_KNEE, R_ANKLE),
    ):
        hip_j = _joint(points, hip_i)
        knee_j = _joint(points, knee_i)
        ankle_j = _joint(points, ankle_i)
        if hip_j is None or knee_j is None or ankle_j is None:
            continue
        thigh = _dist(hip_j, knee_j)
        shin = _dist(knee_j, ankle_j)
        span = _dist(hip_j, ankle_j)
        if thigh + shin > 1.0:
            bends.append(1.0 - span / (thigh + shin))
    bend = max(bends) if bends else 0.0
    reach = 0.0
    for wrist_i in (L_WRIST, R_WRIST):
        wrist = _joint(points, wrist_i)
        if wrist is not None:
            reach = max(reach, _dist(shoulder, wrist) / torso)
    step = 0.0
    left = _joint(points, L_ANKLE)
    right = _joint(points, R_ANKLE)
    if left is not None and right is not None:
        step = abs(left[0] - right[0]) / torso
    drop = 0.0
    if previous is not None and dt > 1e-3:
        drop = (hip[1] - previous["hip_y"]) / torso / dt
    return {
        "upright": upright,
        "drop": drop,
        "step": step,
        "bend": bend,
        "reach": reach,
        "hip_y": hip[1],
        "torso": torso,
    }


def _softmax(logits):
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def raw_scores(measure_now, motion=None) -> list[float]:
    """Eight shares that sum to 1. None means nobody is in frame.

    motion is the 2 second summary. Possible fall uses the peak drop in that
    window, and only after the torso has actually left upright.
    """
    if measure_now is None:
        scores = [0.0] * len(LABELS)
        scores[LABELS.index("no_person")] = 1.0
        return scores
    upright = measure_now["upright"]
    drop = measure_now["drop"]
    step = measure_now["step"]
    bend = measure_now["bend"]
    reach = measure_now["reach"]
    upright_loss = 0.0 if motion is None else motion["upright_loss"]
    peak_drop = drop if motion is None else max(drop, motion["max_drop"])
    logits = [_LOGIT[name] for name in LABELS]
    up = max(upright, 0.0)
    # A wide step or a folded leg takes the share that standing would have had.
    logits[LABELS.index("standing")] += 2.4 * up * max(0.0, 1.0 - max(step - 0.5, 0.0)) * (1.0 - max(bend, 0.0))
    logits[LABELS.index("walking")] += 3.0 * up * min(max(step - 0.5, 0.0), 1.2)
    logits[LABELS.index("sitting")] += 3.4 * up * max(bend - 0.35, 0.0)
    if reach > 1.15 and up > 0.6:
        logits[LABELS.index("using_an_object")] += 5.5 * (reach - 0.9)
        logits[LABELS.index("standing")] *= 0.15
    horizontal = 1.0 - up
    logits[LABELS.index("lying")] += 3.2 * horizontal * (1.0 if peak_drop < 0.8 else 0.3)
    # A crouch drops the hips and stays upright. A fall also loses the torso.
    if upright_loss >= 0.35 and peak_drop > 0.6:
        logits[LABELS.index("possible_fall")] += 2.8 * min(peak_drop, 3.0) * (0.4 + max(horizontal, upright_loss))
    if upright < 0.2 and drop < 0.4:
        logits[LABELS.index("lying")] += 1.2
    logits[LABELS.index("unclear")] += 0.3 * (1.0 - abs(upright))
    return _softmax(logits)
