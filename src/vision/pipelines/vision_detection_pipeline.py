from collections.abc import Iterable, Iterator
from cv2.typing import MatLike
from typing import Callable
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
    HandLandmarkerOptions,
    HandLandmarker,
)
from vision.hand.hand_landmarker import create_hand_landmarker_options
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
    FaceLandmarkerOptions,
    FaceLandmarker
)
from vision.face.face_landmarker import create_face_landmarker_options
from vision.async_landmark_processor import AsyncLandmarkProcessor

type HandLandmarkerHandler = Callable[[HandLandmarkerResult, int], None]
type FaceLandmarkerHandler = Callable[[FaceLandmarkerResult, int], None]


class VisionDetectionPipeline:
    def __init__(
        self,
        on_hand_result: HandLandmarkerHandler,
        on_face_result: FaceLandmarkerHandler,
        on_hand_debug: HandLandmarkerHandler | None = None,
        on_face_debug: FaceLandmarkerHandler | None = None
    ) -> None:
        self._hand_landmarker = AsyncLandmarkProcessor[HandLandmarkerResult, HandLandmarkerOptions](
            HandLandmarker, create_hand_landmarker_options
        )

        self._face_landmarker = AsyncLandmarkProcessor[FaceLandmarkerResult, FaceLandmarkerOptions](
            FaceLandmarker, create_face_landmarker_options
        )

        self._on_hand_result = on_hand_result
        self._on_face_result = on_face_result

        self._on_hand_debug = on_hand_debug
        self._on_face_debug = on_face_debug

    def process(
            self, stream: Iterable[tuple[MatLike, mp.Image, int]]
    ) -> Iterator[MatLike]:
        for frame, mp_image, timestamp_ms in stream:
            self._hand_landmarker.submit(mp_image, timestamp_ms)
            self._face_landmarker.submit(mp_image, timestamp_ms)

            hand_packet = self._hand_landmarker.try_get_result()
            face_packet = self._face_landmarker.try_get_result()

            if hand_packet is not None:
                hand_result, hand_timestamp = hand_packet

                self._on_hand_result(
                    hand_result,
                    hand_timestamp
                )

                if self._on_hand_debug is not None:
                    self._on_hand_debug(hand_result, hand_timestamp)

            if face_packet is not None:
                face_result, face_timestamp = face_packet

                self._latest_face_result = face_packet

                self._on_face_result(
                    face_result,
                    face_timestamp
                )

                if self._on_face_debug is not None:
                    self._on_face_debug(face_result, face_timestamp)

            yield frame

    def close(self) -> None:
        self._hand_landmarker.close()
        self._face_landmarker.close()
