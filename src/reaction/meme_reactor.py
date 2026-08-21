import random
from collections.abc import Iterable
from pathlib import Path

from cv2.typing import MatLike

from vision.types import Gesture

GESTURE_MEMES: dict[Gesture, tuple[str, ...]] = {
    "FIST": ("assets/memes/fist/fist.jpg",),
    "OPEN_PALM": ("assets/memes/open_palm/openpalm.webp",),
    "PEACE": ("assets/memes/peace/peace.jpg",),
    "POINTING": ("assets/memes/pointing/pointing.jpg",),
}


class MemeReactor:
    @staticmethod
    def react(
        stream: Iterable[tuple[MatLike, Gesture | None]],
    ) -> Iterable[tuple[MatLike, Path | None]]:
        for frame, gesture in stream:
            if gesture is None or gesture not in GESTURE_MEMES:
                yield frame, None
                continue

            meme_paths = GESTURE_MEMES[gesture]
            selected_meme_path = random.choice(meme_paths)
            yield frame, Path(selected_meme_path)
