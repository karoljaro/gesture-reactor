import cv2
from cv2.typing import MatLike
from collections.abc import Iterable


class FrameDisplay:
    def __init__(self, window_name: str = "Webcam Feed"):
        self._window_name = window_name

    def show(self, stream: Iterable[MatLike]) -> None:
        cv2.namedWindow(self._window_name)

        for frame in stream:
            cv2.imshow(self._window_name, frame)

            if self._should_close():
                break

    def _should_close(self) -> bool:
        try:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True

            return (
                cv2.getWindowProperty(
                    self._window_name,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            )
        except cv2.error:
            return True

    def close(self) -> None:
        cv2.destroyAllWindows()
