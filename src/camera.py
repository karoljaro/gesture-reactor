from collections.abc import Iterator
from time import monotonic_ns
from threading import Thread, Event
from queue import Queue, Full, Empty
import cv2
from cv2.typing import MatLike


class Camera:
    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._capture = cv2.VideoCapture(device_index)
        self._last_timestamp_ms = -1
        self._stop_event = Event()
        self._frames: Queue[tuple[MatLike, int]] = Queue(maxsize=1)
        self._reader_thread: Thread | None = None
        self._exception: Exception | None = None

    def stream(self) -> Iterator[tuple[MatLike, int]]:
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera with device index {self._device_index}")

        print("Webcam access successfully.")

        self._reader_thread = Thread(target=self._capture_loop)
        self._reader_thread.start()

        try:
            while not self._stop_event.is_set():
                try:
                    frame, timestamp = self._frames.get(timeout=0.1)

                except Empty:
                    continue

                if self._exception is not None:
                    raise self._exception

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

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame, timestamp = self._read_frame_with_timestamp()

                try:
                    self._frames.put_nowait((frame, timestamp))
                except Full:
                    try:
                        self._frames.get_nowait()
                    except Empty:
                        pass

                    self._frames.put_nowait((frame, timestamp))

        except Exception as exc:
            self._exception = exc
            self._stop_event.set()

    def close(self) -> None:
        self._stop_event.set()

        if self._reader_thread is not None:
            self._reader_thread.join()

        self._capture.release()
