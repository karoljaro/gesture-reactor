from pathlib import Path

import cv2


class MemeDisplay:
    def __init__(self, window_name: str = "Meme Display") -> None:
        self._window_name = window_name
        self._is_open = False

    def show(self, meme_path: Path) -> None:
        if meme_path is not None:
            meme_image = cv2.imread(meme_path.as_posix())

            if meme_image is not None:
                if not self._is_open:
                    cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                    self._is_open = True

                cv2.imshow(self._window_name, meme_image)

    def close(self) -> None:
        if not self._is_open:
            return

        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass
        finally:
            self._is_open = False
