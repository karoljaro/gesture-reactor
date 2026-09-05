from collections.abc import Callable


class Stabilizer[T]:
    MAX_TRANSIENT_COUNT = 1
    MIN_REQUIRED_STREAK_COUNT = 5

    def __init__(self) -> None:
        self._transient_count = 0
        self._candidate: T | None = None
        self._candidate_count = 0
        self._stable: T | None = None

    def stabilize(
        self,
        value: T | None,
        is_transient: Callable[[T], bool],
    ) -> T | None:
        if value is None or is_transient(value):
            self._transient_count += 1

            if self._transient_count > self.MAX_TRANSIENT_COUNT:
                self._candidate = None
                self._candidate_count = 0
                self._stable = None

            return self._stable

        self._transient_count = 0

        if value == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = value
            self._candidate_count = 1

        if self._candidate_count >= self.MIN_REQUIRED_STREAK_COUNT:
            self._stable = value

        return self._stable