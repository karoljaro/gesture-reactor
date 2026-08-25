from collections.abc import Iterator, Iterable
import mediapipe as mp
from cv2.typing import MatLike
from vision.async_landmark_processor import AsyncLandmarkProcessor
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
    FaceLandmarkerOptions,
    FaceLandmarker
)
from vision.face.face_landmarker import create_face_landmarker_options


class FaceReactionPipeline:
    def __init__(self) -> None:
        self._face_landmarker = AsyncLandmarkProcessor[FaceLandmarkerResult, FaceLandmarkerOptions](
            FaceLandmarker, create_face_landmarker_options
        )

    def process(
        self, stream: Iterable[tuple[MatLike, mp.Image, int]]
    ) -> Iterator[tuple[MatLike, FaceLandmarkerResult, int]]:
        detection_stream = self._face_landmarker.process(stream)
        yield from detection_stream
