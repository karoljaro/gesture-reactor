import random
from collections.abc import Callable
from pathlib import Path

from vision.face.types.expression import Expression
from vision.hand.types.gesture import Gesture


GESTURE_MEMES: dict[Gesture, tuple[str, ...]] = {
    Gesture.FIST: ("assets/memes/fist/fist.jpg",),
    Gesture.OPEN_PALM: ("assets/memes/open_palm/openpalm.webp",),
    Gesture.PEACE: ("assets/memes/peace/peace.jpg",),
    Gesture.POINTING: ("assets/memes/pointing/pointing.jpg",),
}

EXPRESSION_MEMES: dict[Expression, tuple[str, ...]] = {
    Expression.SMILE: ("assets/memes/expression/smile.jpg",),
    Expression.FROWN: ("assets/memes/expression/frown.jpg",),
    Expression.EYES_CLOSED: ("assets/memes/expression/eyes_closed.jpg",),
    Expression.SURPRISED: ("assets/memes/expression/surprised.jpg",),
}

COMBINED_MEMES: dict[tuple[Gesture, Expression], tuple[str, ...]] = {
    # (Gesture.PEACE, Expression.SMILE): (
    #     "assets/memes/combined/peace_smile.jpg",
    # ),
}


class MemeReactor:
    COMBINATION_MAX_AGE_MS = 800

    def __init__(
        self,
        on_meme: Callable[[Path], None],
    ) -> None:
        self._on_meme = on_meme

        self._latest_gesture: tuple[Gesture, int] | None = None
        self._latest_expression: tuple[Expression, int] | None = None

    def handle_gesture(
        self,
        gesture: Gesture,
        timestamp_ms: int,
    ) -> None:
        self._latest_gesture = gesture, timestamp_ms

        expression = self._get_fresh_expression(timestamp_ms)

        if expression is not None:
            if self._emit(COMBINED_MEMES.get((gesture, expression))):
                return

        self._emit(GESTURE_MEMES.get(gesture))

    def handle_expression(
        self,
        expression: Expression,
        timestamp_ms: int,
    ) -> None:
        self._latest_expression = expression, timestamp_ms

        if expression is Expression.NEUTRAL:
            return

        gesture = self._get_fresh_gesture(timestamp_ms)

        if gesture is not None:
            if self._emit(COMBINED_MEMES.get((gesture, expression))):
                return

        self._emit(EXPRESSION_MEMES.get(expression))

    def _get_fresh_gesture(
        self,
        timestamp_ms: int,
    ) -> Gesture | None:
        if self._latest_gesture is None:
            return None

        gesture, gesture_timestamp = self._latest_gesture

        if abs(timestamp_ms - gesture_timestamp) > self.COMBINATION_MAX_AGE_MS:
            return None

        return gesture

    def _get_fresh_expression(
        self,
        timestamp_ms: int,
    ) -> Expression | None:
        if self._latest_expression is None:
            return None

        expression, expression_timestamp = self._latest_expression

        if expression is Expression.NEUTRAL:
            return None

        if abs(timestamp_ms - expression_timestamp) > self.COMBINATION_MAX_AGE_MS:
            return None

        return expression

    def _emit(
        self,
        memes: tuple[str, ...] | None,
    ) -> bool:
        if not memes:
            return False

        self._on_meme(Path(random.choice(memes)))
        return True
