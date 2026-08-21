from collections.abc import Iterable
from pathlib import Path

import cv2
from cv2.typing import MatLike


class MemeDisplay:
    def show(self, stream: Iterable[tuple[MatLike, Path | None]]) -> Iterable[MatLike]:
        for frame, meme_path in stream:
            if meme_path is not None:
                meme_image = cv2.imread(meme_path.as_posix())

                if meme_image is not None:
                    cv2.namedWindow("Meme Display", cv2.WINDOW_NORMAL)
                    cv2.imshow("Meme Display", meme_image)

            yield frame
