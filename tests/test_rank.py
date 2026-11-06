from roomwatch.labels import LABELS
from roomwatch.rank import Ranker

# COCO indexes used by the stick figure.
L_SHO, R_SHO = 5, 6
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16


def blank():
    return [(0.0, 0.0, 0.0) for _ in range(17)]


def put(points, index, x, y, conf=0.95):
    points[index] = (float(x), float(y), conf)


def standing(hip_y=100.0, lean=0.0, ankle_gap=10.0):
    points = blank()
    put(points, L_SHO, 80 + lean, hip_y - 60)
    put(points, R_SHO, 120 + lean, hip_y - 60)
    put(points, L_HIP, 90, hip_y)
    put(points, R_HIP, 110, hip_y)
    put(points, L_KNEE, 90, hip_y + 40)
    put(points, R_KNEE, 110, hip_y + 40)
    put(points, L_ANK, 100 - ankle_gap, hip_y + 80)
    put(points, R_ANK, 100 + ankle_gap, hip_y + 80)
    return points


def sitting():
    points = blank()
    put(points, L_SHO, 80, 40)
    put(points, R_SHO, 120, 40)
    put(points, L_HIP, 90, 100)
    put(points, R_HIP, 110, 100)
    put(points, L_KNEE, 40, 105)
    put(points, R_KNEE, 160, 105)
    put(points, L_ANK, 95, 110)
    put(points, R_ANK, 105, 110)
    return points


def lying():
    points = blank()
    put(points, L_SHO, 20, 50)
    put(points, R_SHO, 20, 70)
    put(points, L_HIP, 90, 50)
    put(points, R_HIP, 90, 70)
    put(points, L_KNEE, 130, 50)
    put(points, R_KNEE, 130, 70)
    put(points, L_ANK, 170, 50)
    put(points, R_ANK, 170, 70)
    return points


def reaching():
    points = standing()
    put(points, L_WRI, 20, 40)
    put(points, R_WRI, 180, 40)
    return points


def test_shares_sum_to_one():
    ranker = Ranker()
    scores = ranker.update(0.0, standing())
    assert abs(sum(scores) - 1.0) < 1e-6
    assert len(scores) == len(LABELS)


def test_empty_room_is_nobody():
    ranker = Ranker(half_life_s=0.2)
    for step in range(8):
        ranker.update(step * 0.25, blank())
    assert ranker.leader() == "no_person"


def test_held_stand_climbs():
    ranker = Ranker(half_life_s=0.6)
    first = ranker.update(0.0, standing())[LABELS.index("standing")]
    later = None
    for step in range(1, 16):
        later = ranker.update(step * 0.25, standing())[LABELS.index("standing")]
    assert later > first
    assert later > 0.7
    assert ranker.leader() == "standing"


def test_wide_step_while_upright_is_walking():
    ranker = Ranker(half_life_s=0.3)
    for step in range(12):
        ranker.update(step * 0.25, standing(ankle_gap=40))
    assert ranker.leader() == "walking"


def test_folded_legs_are_sitting():
    ranker = Ranker(half_life_s=0.3)
    for step in range(12):
        ranker.update(step * 0.25, sitting())
    assert ranker.leader() == "sitting"


def test_horizontal_body_is_lying():
    ranker = Ranker(half_life_s=0.3)
    for step in range(12):
        ranker.update(step * 0.25, lying())
    assert ranker.leader() == "lying"


def tipped(hip_y, above):
    """above is how many pixels the shoulders sit over the hips. 0 is flat."""
    points = standing(hip_y=hip_y)
    put(points, L_SHO, 20, hip_y - above)
    put(points, R_SHO, 40, hip_y - above)
    return points


def test_fast_drop_leads_with_possible_fall():
    ranker = Ranker(half_life_s=0.25)
    ranker.update(0.0, standing())
    ranker.update(0.25, tipped(160, 25))
    view = ranker.update(0.5, tipped(220, 4))
    assert ranker.leader() == "possible_fall"
    assert view[LABELS.index("possible_fall")] > 0.4


def test_crouch_stays_off_the_fall_bar():
    ranker = Ranker(half_life_s=0.25)
    ranker.update(0.0, standing())
    for step, hip in enumerate((150, 190, 220), start=1):
        ranker.update(step * 0.25, standing(hip_y=hip))
    assert ranker.leader() != "possible_fall"


def test_fall_stays_for_the_two_second_window():
    ranker = Ranker(half_life_s=0.2)
    ranker.update(0.0, standing())
    ranker.update(0.25, tipped(160, 25))
    ranker.update(0.5, tipped(220, 4))
    for step in range(4):
        ranker.update(0.75 + step * 0.25, lying())
    assert ranker.leader() == "possible_fall"
    ranker.update(3.2, lying())
    ranker.update(3.45, lying())
    assert ranker.leader() == "lying"


def test_slow_lie_does_not_outrank_lying():
    ranker = Ranker(half_life_s=0.3)
    ranker.update(0.0, standing())
    for step in range(1, 10):
        ranker.update(step * 0.5, lying())
    assert ranker.leader() == "lying"


def test_reach_can_lead_using_an_object():
    ranker = Ranker(half_life_s=0.25)
    for step in range(14):
        ranker.update(step * 0.25, reaching())
    assert ranker.leader() == "using_an_object"
