import cv2
from cv2.typing import MatLike
from collections.abc import Iterable


class VisionProcessor:
    def process(self, stream: Iterable[MatLike]) -> Iterable[MatLike]:
        for frame in stream:
            flipped_frame = cv2.flip(frame, 1)
            yield flipped_frame
