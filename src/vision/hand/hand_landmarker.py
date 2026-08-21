import cv2
import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks.python.vision import HandLandmarkerResult


class HandLandmarker:
    HAND_CONNECTIONS = [
        # Nadgarstek i kciuk
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        # Palec wskazujący
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        # Palec środkowy
        (9, 10),
        (10, 11),
        (11, 12),
        # Palec serdeczny
        (13, 14),
        (14, 15),
        (15, 16),
        # Mały palec
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        # Połączenia między podstawami palców
        (5, 9),
        (9, 13),
        (13, 17),
    ]

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

    def draw(self, frame: MatLike, result: HandLandmarkerResult | None) -> MatLike:
        if result is None:
            return frame

        height, width = frame.shape[:2]

        for hand in result.hand_landmarks:
            for start_idx, end_idx in self.HAND_CONNECTIONS:
                start_landmark = hand[start_idx]
                end_landmark = hand[end_idx]

                start_x = int(start_landmark.x * width)
                start_y = int(start_landmark.y * height)
                end_x = int(end_landmark.x * width)
                end_y = int(end_landmark.y * height)

                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            for landmark in hand:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        return frame

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(
        self, result: HandLandmarkerResult, _output_image: mp.Image, _timestamp_ms: int
    ) -> None:
        self._latest_result = result
