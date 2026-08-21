from collections.abc import Iterable, Iterator
from queue import Empty, Full, Queue

import cv2
import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks.python.vision import HandLandmarkerResult


class HandLandmarker:
    def __init__(self) -> None:
        base_options = mp.tasks.BaseOptions
        hand_landmarker = mp.tasks.vision.HandLandmarker
        hand_landmarker_options = mp.tasks.vision.HandLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode

        self._results: Queue[tuple[mp.Image, HandLandmarkerResult, int]] = Queue(maxsize=1)

        options = hand_landmarker_options(
            base_options=base_options(
                model_asset_path="models/hand_landmarker.task",
            ),
            running_mode=running_mode.LIVE_STREAM,
            result_callback=self._on_result,
        )

        self._landmarker = hand_landmarker.create_from_options(options)

    def process(
        self, stream: Iterable[tuple[MatLike, int]]
    ) -> Iterator[tuple[MatLike, HandLandmarkerResult, int]]:
        for frame, timestamp in stream:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            self._landmarker.detect_async(mp_image, timestamp)

            try:
                output_image, result, result_timestamp = self._results.get_nowait()
            except Empty:
                continue

            output_frame = cv2.cvtColor(output_image.numpy_view(), cv2.COLOR_RGB2BGR)
            yield output_frame, result, result_timestamp

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(
        self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
    ) -> None:
        packet = output_image, result, timestamp_ms

        try:
            self._results.put_nowait(packet)
        except Full:
            try:
                self._results.get_nowait()
            except Empty:
                pass

            self._results.put_nowait(packet)
