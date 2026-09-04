from collections.abc import Callable


class Stabilizer[T]:
    MAX_TRANSIENT_COUNT = 1
    MIN_REQUIRED_STREAK_COUNT = 5

    def __init__(self) -> None:
        self._transient_count: int = 0
        self._candidate: T | None = None
        self._candidate_count: int = 0
        self._last_emitted: T | None = None

    def stabilize(
        self, value: T | None, is_transient: Callable[[T], bool]
    ) -> T | None:
        if value is None:
            return None

        if is_transient(value):
            self._transient_count += 1

            if self._transient_count > self.MAX_TRANSIENT_COUNT:
                self._reset_candidate()

            return None

        self._transient_count = 0

        if value == self._candidate:
            self._candidate_count += 1

            if (
                self._candidate_count >= self.MIN_REQUIRED_STREAK_COUNT
                and value != self._last_emitted
            ):
                self._last_emitted = value
                return value

        else:
            self._reset_candidate()
            self._candidate = value
            self._candidate_count = 1

        return None

    def _reset_candidate(self) -> None:
        self._candidate = None
        self._candidate_count = 0
        self._transient_count = 0
