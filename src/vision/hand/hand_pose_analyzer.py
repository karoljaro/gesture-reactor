from math import acos, degrees, sqrt
from typing import Literal

from mediapipe.tasks.python.components.containers.landmark import Landmark
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

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

FINGER_LANDMARKS: dict[FingerName, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


class HandPoseAnalyzer:
    def analyze(
        self,
        latest_result: HandLandmarkerResult | None,
    ) -> dict[FingerName, FingerState] | None:
        if latest_result is None or not latest_result.hand_world_landmarks:
            return None

        hand_landmarks = latest_result.hand_world_landmarks[0]
        finger_states: dict[FingerName, FingerState] = {}

        for finger, landmarks in FINGER_LANDMARKS.items():
            if finger == "thumb":
                primary_thresholds = FINGER_ANGLE_THRESHOLDS[finger]["mcp"]
                secondary_thresholds = FINGER_ANGLE_THRESHOLDS[finger]["ip"]
            else:
                primary_thresholds = FINGER_ANGLE_THRESHOLDS[finger]["pip"]
                secondary_thresholds = FINGER_ANGLE_THRESHOLDS[finger]["dip"]

            primary_angle = self._calculate_angle(
                hand_landmarks[landmarks[0]],
                hand_landmarks[landmarks[1]],
                hand_landmarks[landmarks[2]],
            )
            if primary_angle is None:
                return None

            primary_state = self._classify_angle(primary_angle, primary_thresholds)
            if primary_state is not None:
                finger_states[finger] = primary_state
                continue

            secondary_angle = self._calculate_angle(
                hand_landmarks[landmarks[1]],
                hand_landmarks[landmarks[2]],
                hand_landmarks[landmarks[3]],
            )
            if secondary_angle is None:
                return None

            secondary_state = self._classify_angle(secondary_angle, secondary_thresholds)
            finger_states[finger] = secondary_state if secondary_state is not None else "PARTIAL"

        return finger_states

    @staticmethod
    def _calculate_angle(
        first: Landmark,
        middle: Landmark,
        last: Landmark,
    ) -> float | None:
        first_x = first.x - middle.x
        first_y = first.y - middle.y
        first_z = first.z - middle.z

        last_x = last.x - middle.x
        last_y = last.y - middle.y
        last_z = last.z - middle.z

        dot_product = first_x * last_x + first_y * last_y + first_z * last_z

        first_length = sqrt(first_x * first_x + first_y * first_y + first_z * first_z)
        last_length = sqrt(last_x * last_x + last_y * last_y + last_z * last_z)
        lengths_product = first_length * last_length

        if lengths_product == 0.0:
            return None

        cosine = dot_product / lengths_product
        cosine = max(-1.0, min(1.0, cosine))

        return degrees(acos(cosine))

    @staticmethod
    def _classify_angle(
        angle: float,
        thresholds: dict[AngleThresholdState, float],
    ) -> FingerState | None:
        if angle <= thresholds["folded"]:
            return "FOLDED"

        if angle >= thresholds["extended"]:
            return "EXTENDED"

        return None
