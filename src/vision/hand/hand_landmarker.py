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

        self._latest_result: HandLandmarkerResult | None = None

        options = hand_landmarker_options(
            base_options=base_options(
                model_asset_path="models/hand_landmarker.task",
            ),
            running_mode=running_mode.LIVE_STREAM,
            result_callback=self._on_result,
        )

        self._landmarker = hand_landmarker.create_from_options(options)

    @property
    def latest_result(self) -> HandLandmarkerResult | None:
        return self._latest_result

    def detect(self, frame: MatLike, timestamp: int) -> MatLike:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._landmarker.detect_async(mp_image, timestamp)

        return frame

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(
        self, result: HandLandmarkerResult, _output_image: mp.Image, _timestamp_ms: int
    ) -> None:
        self._latest_result = result
