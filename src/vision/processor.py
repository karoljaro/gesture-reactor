from collections.abc import Iterable

import cv2
from cv2.typing import MatLike


class VisionProcessor:
    def process(self, stream: Iterable[tuple[MatLike, int]]) -> Iterable[tuple[MatLike, int]]:
        for frame, timestamp in stream:
            flipped_frame = cv2.flip(frame, 1)
            yield flipped_frame, timestamp
