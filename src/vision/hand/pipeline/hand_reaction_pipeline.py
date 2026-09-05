from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from vision.hand.hand_gesture_classifier import HandGestureClassifier
from vision.classification_stabilizer import Stabilizer
from vision.hand.hand_pose_analyzer import HandPoseAnalyzer
from vision.types import ClassifierResult, Gesture

from collections.abc import Callable


class HandReactionPipeline:
    def __init__(
        self,
        on_gesture: Callable[[Gesture, int], None]
    ) -> None:
        self._pose_analyzer = HandPoseAnalyzer()
        self._classifier = HandGestureClassifier()
        self._stabilizer = Stabilizer[ClassifierResult]()
        self._on_gesture = on_gesture

    def handle_result(
        self,
        result: HandLandmarkerResult,
        timestamp_ms: int
    ) -> None:
        finger_states = self._pose_analyzer.analyze(result)
        classified_gesture = self._classifier.classify(finger_states)
        stabilized_gesture = self._stabilizer.stabilize(
            classified_gesture,
            lambda gesture: gesture == "UNKNOWN"
        )

        if stabilized_gesture is None or stabilized_gesture == "UNKNOWN":
            return

        self._on_gesture(
            stabilized_gesture,
            timestamp_ms,
        )
