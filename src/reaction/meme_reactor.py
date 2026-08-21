import random
from pathlib import Path

from vision.types import Gesture

GESTURE_MEMES: dict[Gesture, tuple[str, ...]] = {
    "FIST": ("assets/memes/fist/fist.jpg",),
    "OPEN_PALM": ("assets/memes/open_palm/openpalm.webp",),
    "PEACE": ("assets/memes/peace/peace.jpg",),
    "POINTING": ("assets/memes/pointing/pointing.jpg",),
}


class MemeReactor:
    @staticmethod
    def react(gesture: Gesture | None) -> Path | None:
        if gesture is None or gesture not in GESTURE_MEMES:
            return None

        meme_paths = GESTURE_MEMES[gesture]
        selected_meme_path = random.choice(meme_paths)
        return Path(selected_meme_path)
