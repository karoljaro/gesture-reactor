import random
from collections.abc import Callable
from pathlib import Path

from vision.face.types.expression import Expression
from vision.types import Gesture

type ReactionKey = Gesture | Expression | tuple[Gesture, Expression]

type MemeHandler = Callable[[Path], None]


GESTURE_MEMES: dict[Gesture, tuple[str, ...]] = {
    "FIST": ("assets/memes/fist/fist.jpg",),
    "OPEN_PALM": ("assets/memes/open_palm/openpalm.webp",),
    "PEACE": ("assets/memes/peace/peace.jpg",),
    "POINTING": ("assets/memes/pointing/pointing.jpg",),
}

EXPRESSION_MEMES: dict[Expression, tuple[str, ...]] = {
    Expression.SMILE: ("assets/memes/expression/smile.jpg",),
    Expression.FROWN: ("assets/memes/expression/frown.jpg",),
    Expression.EYES_CLOSED: ("assets/memes/expression/eyes_closed.jpg",),
    Expression.SURPRISED: ("assets/memes/expression/surprised.jpg",),
}


def _combo(
    gesture: Gesture,
    expression: Expression,
) -> tuple[Gesture, Expression]:
    return gesture, expression


COMBINED_MEMES: dict[tuple[Gesture, Expression], tuple[str, ...]] = {
    _combo("PEACE", Expression.SMILE): ("assets/memes/combo/peace_smile/first.png",),
    _combo("FIST", Expression.FROWN): ("assets/memes/combo/fist_frown/first.png",),
    _combo("POINTING", Expression.SURPRISED): ("assets/memes/combo/pointing_surprised/first.png",),
}


class MemeReactor:
    COMBINATION_MAX_AGE_MS = 800
    REACTION_COOLDOWN_MS = 800

    def __init__(
        self,
        on_meme: MemeHandler,
    ) -> None:
        self._on_meme = on_meme

        self._latest_gesture: tuple[Gesture, int] | None = None
        self._latest_expression: tuple[Expression, int] | None = None

        self._last_reaction: ReactionKey | None = None
        self._last_reaction_timestamp: int = -1

    def handle_gesture(
        self,
        gesture: Gesture,
        timestamp_ms: int,
    ) -> None:
        self._latest_gesture = gesture, timestamp_ms

        expression = self._get_fresh_expression(timestamp_ms)

        if expression is not None:
            combo = _combo(gesture, expression)

            if self._emit(
                combo,
                COMBINED_MEMES.get(combo),
                timestamp_ms,
            ):
                return

        self._emit(
            gesture,
            GESTURE_MEMES.get(gesture),
            timestamp_ms,
        )

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
            combo = _combo(gesture, expression)

            if self._emit(
                combo,
                COMBINED_MEMES.get(combo),
                timestamp_ms,
            ):
                return

        self._emit(
            expression,
            EXPRESSION_MEMES.get(expression),
            timestamp_ms,
        )

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
        reaction: ReactionKey,
        meme_paths: tuple[str, ...] | None,
        timestamp_ms: int,
    ) -> bool:
        if not meme_paths:
            return False

        if (
            reaction == self._last_reaction
            and timestamp_ms - self._last_reaction_timestamp < self.REACTION_COOLDOWN_MS
        ):
            self._last_reaction_timestamp = timestamp_ms
            return True

        self._last_reaction = reaction
        self._last_reaction_timestamp = timestamp_ms

        self._on_meme(Path(random.choice(meme_paths)))

        return True
