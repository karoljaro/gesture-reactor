import random
from collections.abc import Callable
from vision.types import Gesture
from pathlib import Path
from vision.face.types.expression import Expression

GESTURE_MEMES: dict[Gesture, tuple[str, ...]] = {
    "FIST": ("assets/memes/fist/fist.jpg",),
    "OPEN_PALM": ("assets/memes/open_palm/openpalm.webp",),
    "PEACE": ("assets/memes/peace/peace.jpg",),
    "POINTING": ("assets/memes/pointing/pointing.jpg",),
}

type HandleGesture = Callable[[Path, int], None]


class MemeReactor:
    def __init__(
        self,
        on_meme: Callable[[Path], None]
    ) -> None:
        self._on_meme = on_meme

    def handle_gesture(
        self,
        gesture: Gesture,
        _timestamp_ms: int,
    ) -> None:
        meme_path = self._get_meme_path(gesture)

        if meme_path is not None:
            self._on_meme(meme_path)

    def handle_expression(
        self,
        expression: Expression,
        _timestamp_ms: int
    ) -> None:
        print(expression)

    def _get_meme_path(self, gesture: Gesture) -> Path | None:
        if gesture is None or gesture not in GESTURE_MEMES:
            return None

        meme_paths = GESTURE_MEMES[gesture]
        selected_meme_path = Path(random.choice(meme_paths))
        return selected_meme_path
