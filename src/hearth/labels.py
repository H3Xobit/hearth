"""The eight bars. Order is the storage order. The screen sorts by score."""

LABELS = (
    "standing",
    "walking",
    "sitting",
    "using_an_object",
    "unclear",
    "no_person",
    "lying",
    "possible_fall",
)

STANDING = LABELS.index("standing")
WALKING = LABELS.index("walking")
SITTING = LABELS.index("sitting")
USING_AN_OBJECT = LABELS.index("using_an_object")
UNCLEAR = LABELS.index("unclear")
NO_PERSON = LABELS.index("no_person")
LYING = LABELS.index("lying")
POSSIBLE_FALL = LABELS.index("possible_fall")

# A recovered person is standing or walking. Sitting does not cancel a fall hold.
RECOVERED = frozenset({STANDING, WALKING})
