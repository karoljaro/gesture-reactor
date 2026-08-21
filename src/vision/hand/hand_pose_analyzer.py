from typing import Literal

import numpy as np
from mediapipe.tasks.python.vision import HandLandmarkerResult

from ..types import FingerName, FingerState

FingerPoint = Literal["pip", "dip", "mcp", "ip"]
AngleThresholdState = Literal["folded", "extended"]

FINGER_ANGLE_THRESHOLDS: dict[FingerName, dict[FingerPoint, dict[AngleThresholdState, float]]] = {
    "thumb": {
        "mcp": {
            "folded": 158.0,
            "extended": 162.0,
        },
        "ip": {
            "folded": 145.0,
            "extended": 155.0,
        },
    },
    "index": {
        "pip": {
            "folded": 105.0,
            "extended": 140.0,
        },
        "dip": {
            "folded": 115.0,
            "extended": 145.0,
        },
    },
    "middle": {
        "pip": {
            "folded": 110.0,
            "extended": 140.0,
        },
        "dip": {
            "folded": 115.0,
            "extended": 145.0,
        },
    },
    "ring": {
        "pip": {
            "folded": 115.0,
            "extended": 140.0,
        },
        "dip": {
            "folded": 120.0,
            "extended": 145.0,
        },
    },
    "pinky": {
        "pip": {
            "folded": 120.0,
            "extended": 150.0,
        },
        "dip": {
            "folded": 115.0,
            "extended": 150.0,
        },
    },
}


class HandPoseAnalyzer:
    def __init__(self) -> None:
        self._FINGER_LANDMARKS: dict[FingerName, tuple[int, int, int, int]] = {
            "thumb": (1, 2, 3, 4),
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }

    def analyze(
        self,
        latest_result: HandLandmarkerResult | None,
    ) -> dict[FingerName, FingerState] | None:
        if latest_result is None or not latest_result.hand_world_landmarks:
            return None

        finger_angles: dict[FingerName, FingerState] = {}

        for finger, landmarks in self._FINGER_LANDMARKS.items():
            if finger == "thumb":
                mcp_angle = self._calculate_angle(
                    landmarks[0], landmarks[1], landmarks[2], latest_result
                )
                ip_angle = self._calculate_angle(
                    landmarks[1], landmarks[2], landmarks[3], latest_result
                )

                if mcp_angle is None or ip_angle is None:
                    return None

                finger_angles[finger] = self._classify_thumb(
                    mcp_angle,
                    ip_angle,
                )
                continue

            pip_angle = self._calculate_angle(
                landmarks[0], landmarks[1], landmarks[2], latest_result
            )
            dip_angle = self._calculate_angle(
                landmarks[1], landmarks[2], landmarks[3], latest_result
            )

            if pip_angle is None or dip_angle is None:
                return None

            finger_angles[finger] = self._classify_finger(
                finger,
                pip_angle,
                dip_angle,
            )

        return finger_angles

    def _classify_thumb(self, mcp_angle: float, ip_angle: float) -> FingerState:
        thumb_thresholds = FINGER_ANGLE_THRESHOLDS["thumb"]

        if mcp_angle <= thumb_thresholds["mcp"]["folded"]:
            return "FOLDED"

        if mcp_angle >= thumb_thresholds["mcp"]["extended"]:
            return "EXTENDED"

        if ip_angle <= thumb_thresholds["ip"]["folded"]:
            return "FOLDED"

        if ip_angle >= thumb_thresholds["ip"]["extended"]:
            return "EXTENDED"

        return "PARTIAL"

    def _classify_finger(
        self, finger: FingerName, pip_angle: float, dip_angle: float
    ) -> FingerState:
        finger_thresholds = FINGER_ANGLE_THRESHOLDS[finger]

        if pip_angle <= finger_thresholds["pip"]["folded"]:
            return "FOLDED"

        if pip_angle >= finger_thresholds["pip"]["extended"]:
            return "EXTENDED"

        if dip_angle <= finger_thresholds["dip"]["folded"]:
            return "FOLDED"

        if dip_angle >= finger_thresholds["dip"]["extended"]:
            return "EXTENDED"

        return "PARTIAL"

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
