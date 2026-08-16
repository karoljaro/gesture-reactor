import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult


class HandPoseAnalyzer:
    def __init__(self) -> None:
        self._FINGER_LANDMARKS = {
            # "thumb": (1, 2, 3, 4),
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }

    def analyze(
        self,
        latest_result: HandLandmarkerResult | None,
    ) -> dict[str, tuple[float, float]] | None:
        if latest_result is None or not latest_result.hand_world_landmarks:
            return None

        finger_angles: dict[str, tuple[float, float]] = {}

        for finger, landmarks in self._FINGER_LANDMARKS.items():
            first_angle = self._calculate_angle(
                landmarks[0],
                landmarks[1],
                landmarks[2],
                latest_result,
            )

            second_angle = self._calculate_angle(
                landmarks[1],
                landmarks[2],
                landmarks[3],
                latest_result,
            )

            if first_angle is None or second_angle is None:
                return None

            finger_angles[finger] = (
                first_angle,
                second_angle,
            )

        return finger_angles

    def _calculate_angle(
        self,
        landmark1: int,
        middle_landmark: int,
        landmark3: int,
        latest_result: HandLandmarkerResult | None,
    ) -> float | None:
        if latest_result is None or not latest_result.hand_world_landmarks:
            return None

        landmark_1_points = np.array(
            [
                latest_result.hand_world_landmarks[0][landmark1].x,
                latest_result.hand_world_landmarks[0][landmark1].y,
                latest_result.hand_world_landmarks[0][landmark1].z,
            ]
        )

        middle_landmark_points = np.array(
            [
                latest_result.hand_world_landmarks[0][middle_landmark].x,
                latest_result.hand_world_landmarks[0][middle_landmark].y,
                latest_result.hand_world_landmarks[0][middle_landmark].z,
            ]
        )

        landmark_3_points = np.array(
            [
                latest_result.hand_world_landmarks[0][landmark3].x,
                latest_result.hand_world_landmarks[0][landmark3].y,
                latest_result.hand_world_landmarks[0][landmark3].z,
            ]
        )

        v_1_m = landmark_1_points - middle_landmark_points
        v_m_3 = landmark_3_points - middle_landmark_points

        dot_prod = np.dot(v_1_m, v_m_3)

        norms_prod = np.linalg.norm(v_1_m) * np.linalg.norm(v_m_3)

        if norms_prod == 0:
            return None

        cos_angle = dot_prod / norms_prod

        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle_rad = np.arccos(cos_angle)

        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)
