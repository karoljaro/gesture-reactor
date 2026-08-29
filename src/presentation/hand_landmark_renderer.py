import cv2
from cv2.typing import MatLike
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
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
)


class HandLandmarkRenderer:
    def draw(
        self,
        frame: MatLike,
        result: HandLandmarkerResult
    ) -> MatLike | None:
        if result is None:
            return None

        height, width = frame.shape[:2]

        for hand in result.hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
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
