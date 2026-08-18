import cv2
from cv2.typing import MatLike
from collections.abc import Iterator


class Camera:
    def __init__(self, device_index: int = 0, window_name: str = "Webcam Feed"):
        self._device_index = device_index
        self._window_name = window_name
        self._capture = cv2.VideoCapture(device_index)

    def execute(self) -> Iterator[tuple[MatLike, float]]:
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera with device index {self._device_index}")

        print("Webcam access successfully.")

        try:
            while True:
                frame, timestamp = self._read_frame_with_timestamp()

                yield frame, timestamp

        finally:
            self.close()

    def _read_frame_with_timestamp(self) -> tuple[MatLike, float]:
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        timestamp_ms = int(self._capture.get(cv2.CAP_PROP_POS_MSEC))

        return frame, timestamp_ms

    def close(self) -> None:
        self._capture.release()
