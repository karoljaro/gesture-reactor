import cv2
from cv2.typing import MatLike
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
    HandLandmarksConnections,
)


class HandLandmarkRenderer:
    def __init__(self) -> None:
        self._connections = tuple(
            (connection.start, connection.end)
            for connection in HandLandmarksConnections.HAND_CONNECTIONS
        )

    def draw(
        self,
        frame: MatLike,
        result: HandLandmarkerResult,
    ) -> MatLike:
        height, width = frame.shape[:2]

        for hand in result.hand_landmarks:
            points = [
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                )
                for landmark in hand
            ]

            drawn_landmarks = [False] * len(points)

            for start_idx, end_idx in self._connections:
                start_point = points[start_idx]
                end_point = points[end_idx]

                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

                if not drawn_landmarks[start_idx]:
                    cv2.circle(frame, start_point, 5, (0, 255, 0), -1)
                    drawn_landmarks[start_idx] = True

                if not drawn_landmarks[end_idx]:
                    cv2.circle(frame, end_point, 5, (0, 255, 0), -1)
                    drawn_landmarks[end_idx] = True

        return frame
