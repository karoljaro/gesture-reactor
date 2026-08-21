from .types import ClassifierResult, FingerName, FingerState, Gesture

FINGER_ORDER: tuple[FingerName, ...] = ("thumb", "index", "middle", "ring", "pinky")

GESTURE_PATTERNS: dict[Gesture, dict[FingerName, FingerState]] = {
    "FIST": {
        "thumb": "FOLDED",
        "index": "FOLDED",
        "middle": "FOLDED",
        "ring": "FOLDED",
        "pinky": "FOLDED",
    },
    "OPEN_PALM": {
        "thumb": "EXTENDED",
        "index": "EXTENDED",
        "middle": "EXTENDED",
        "ring": "EXTENDED",
        "pinky": "EXTENDED",
    },
    "POINTING": {
        "thumb": "FOLDED",
        "index": "EXTENDED",
        "middle": "FOLDED",
        "ring": "FOLDED",
        "pinky": "FOLDED",
    },
    "PEACE": {
        "thumb": "FOLDED",
        "index": "EXTENDED",
        "middle": "EXTENDED",
        "ring": "FOLDED",
        "pinky": "FOLDED",
    },
}

GESTURE_LOOKUP: dict[tuple[FingerState, ...], Gesture] = {
    tuple(pattern[finger] for finger in FINGER_ORDER): gesture
    for gesture, pattern in GESTURE_PATTERNS.items()
}


class GestureClassifier:
    def classify_gesture(
        self, fingers: dict[FingerName, FingerState] | None
    ) -> ClassifierResult | None:

        if fingers is None:
            return None

        return GESTURE_LOOKUP.get(
            (
                fingers["thumb"],
                fingers["index"],
                fingers["middle"],
                fingers["ring"],
                fingers["pinky"],
            ),
            "UNKNOWN",
        )
