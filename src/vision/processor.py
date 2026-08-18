import cv2
from cv2.typing import MatLike
from collections.abc import Iterable


class VisionProcessor:
    def process(self, stream: Iterable[tuple[MatLike, float]]) -> Iterable[tuple[MatLike, float]]:
        for frame, timestamp in stream:
            flipped_frame = cv2.flip(frame, 1)
            yield flipped_frame, timestamp
