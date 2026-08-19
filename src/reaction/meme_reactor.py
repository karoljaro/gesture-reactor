import random
from pathlib import Path
from vision.gesture_classifier import ClassifierResult

GESTURE_MEMES: dict[ClassifierResult, tuple[str, ...]] = {
    "FIST": (
        "assets/memes/fist/fist.jpg",
    ),
    "OPEN_PALM": (
        "assets/memes/open_palm/openpalm.webp",
    ),
    "PEACE": (
        "assets/memes/peace/peace.jpg",
    ),
    "POINTING": (
        "assets/memes/pointing/pointing.jpg",
    ),
}


class MemeReactor:
    @staticmethod
    def react(gesture: ClassifierResult | None) -> Path | None:
        if gesture is None or gesture not in GESTURE_MEMES:
            return None

        meme_paths = GESTURE_MEMES[gesture]
        selected_meme_path = random.choice(meme_paths)
        return Path(selected_meme_path)
