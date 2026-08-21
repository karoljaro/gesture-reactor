from collections.abc import Iterable

import cv2
import mediapipe as mp
from cv2.typing import MatLike


class VisionProcessor:
    def process(
        self, stream: Iterable[tuple[MatLike, int]]
    ) -> Iterable[tuple[MatLike, mp.Image, int]]:
        for frame, timestamp in stream:
            flipped_frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            yield flipped_frame, mp_image, timestamp
