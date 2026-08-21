from ..types import ClassifierResult, Gesture


class HandGestureStabilizer:
    MAX_UNKNOWN_COUNT = 1
    MIN_REQUIRED_STREAK_COUNT = 5

    def __init__(self) -> None:
        self._unknown_count: int = 0
        self._candidate_gesture: Gesture | None = None
        self._candidate_count: int = 0
        self._last_emitted_gesture: Gesture | None = None

    def update(self, gesture: ClassifierResult | None) -> Gesture | None:
        if gesture is None:
            return None

        if gesture == "UNKNOWN":
            self._unknown_count += 1

            if self._unknown_count > self.MAX_UNKNOWN_COUNT:
                self._reset_candidate()

            return None

        self._unknown_count = 0

        if gesture == self._candidate_gesture:
            self._candidate_count += 1

            if (
                self._candidate_count >= self.MIN_REQUIRED_STREAK_COUNT
                and gesture != self._last_emitted_gesture
            ):
                self._last_emitted_gesture = gesture
                return gesture

        else:
            self._reset_candidate()
            self._candidate_gesture = gesture
            self._candidate_count = 1

        return None

    def _reset_candidate(self) -> None:
        self._candidate_gesture = None
        self._candidate_count = 0
        self._unknown_count = 0
