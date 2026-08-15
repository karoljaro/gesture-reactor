import cv2
from cv2.typing import MatLike
from collections.abc import Iterator


class Camera:
    def __init__(self, device_index: int = 0, window_name: str = "Webcam Feed"):
        self._device_index = device_index
        self._window_name = window_name
        self._capture = cv2.VideoCapture(device_index)

    def execute(self) -> Iterator[MatLike]:
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera with device index {self._device_index}")

        print("Webcam access successfully.")

        try:
            while True:
                frame = self._read_frame()

                yield frame

        finally:
            self.close()

    def _read_frame(self) -> MatLike:
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def close(self) -> None:
        self._capture.release()
