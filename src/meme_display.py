from pathlib import Path
import cv2


class MemeDisplay:
    def show(self, meme_path: Path | None) -> None:
        if meme_path is None:
            return

        meme_image = cv2.imread(meme_path.as_posix())

        if meme_image is None:
            return

        cv2.namedWindow("Meme Display", cv2.WINDOW_NORMAL)
        cv2.imshow("Meme Display", meme_image)
