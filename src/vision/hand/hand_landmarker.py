from collections.abc import Iterable, Iterator
from queue import Empty, Full, Queue

import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks.python.vision import HandLandmarkerResult


class HandLandmarker:
    def __init__(self) -> None:
        base_options = mp.tasks.BaseOptions
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        self._results: Queue[tuple[HandLandmarkerResult, int]] = Queue(maxsize=1)

        options = hand_landmarker_options(
            base_options=base_options(
                model_asset_path="models/hand_landmarker.task",
            ),
            running_mode=running_mode.LIVE_STREAM,
            result_callback=self._on_result,
        )

        self._landmarker = hand_landmarker.create_from_options(options)

    def process(
        self, stream: Iterable[tuple[MatLike, mp.Image, int]]
    ) -> Iterator[tuple[MatLike, HandLandmarkerResult, int]]:
        pending_frames: dict[int, MatLike] = {}

        for frame, mp_image, timestamp in stream:
            pending_frames[timestamp] = frame
            self._landmarker.detect_async(mp_image, timestamp)

            try:
                result, result_timestamp = self._results.get_nowait()
            except Empty:
                continue

            result_frame = pending_frames.pop(result_timestamp)
            stale_timestamps = [
                pending_timestamp
                for pending_timestamp in pending_frames
                if pending_timestamp < result_timestamp
            ]
            for stale_timestamp in stale_timestamps:
                del pending_frames[stale_timestamp]

            yield result_frame, result, result_timestamp

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(
        self, result: HandLandmarkerResult, _output_image: mp.Image, timestamp_ms: int
    ) -> None:
        packet = result, timestamp_ms

        try:
            self._results.put_nowait(packet)
        except Full:
            try:
                self._results.get_nowait()
            except Empty:
                pass

            self._results.put_nowait(packet)
