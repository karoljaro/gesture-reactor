from collections.abc import Iterator
from time import monotonic_ns

import cv2
from cv2.typing import MatLike


class Camera:
    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._capture = cv2.VideoCapture(device_index)
        self._last_timestamp_ms = -1

    def stream(self) -> Iterator[tuple[MatLike, int]]:
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera with device index {self._device_index}")

        print("Webcam access successfully.")

        try:
            while True:
                frame, timestamp = self._read_frame_with_timestamp()

                yield frame, timestamp

        finally:
            self.close()

    def _read_frame_with_timestamp(self) -> tuple[MatLike, int]:
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        timestamp_ms = max(
            monotonic_ns() // 1_000_000,
            self._last_timestamp_ms + 1,
        )
        self._last_timestamp_ms = timestamp_ms

        return frame, timestamp_ms

    def close(self) -> None:
        self._capture.release()
