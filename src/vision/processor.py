import cv2
from cv2.typing import MatLike


class VisionProcessor:
    def process(self, frame: MatLike) -> MatLike:
        flipped_frame = cv2.flip(frame, 1)

        return flipped_frame
