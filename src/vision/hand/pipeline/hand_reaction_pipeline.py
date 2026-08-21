from collections.abc import Iterable

from cv2.typing import MatLike
from mediapipe.tasks.python.vision import HandLandmarkerResult

from vision.hand.hand_gesture_classifier import HandGestureClassifier
from vision.hand.hand_gesture_stabilizer import HandGestureStabilizer
from vision.hand.hand_landmarker import HandLandmarker
from vision.hand.hand_pose_analyzer import HandPoseAnalyzer
from vision.types import Gesture


class HandReactionPipeline:
    def __init__(self) -> None:
        self._hand_landmarker = HandLandmarker()
        self._pose_analyzer = HandPoseAnalyzer()
        self._classifier = HandGestureClassifier()
        self._stabilizer = HandGestureStabilizer()

    def process(
        self, stream: Iterable[tuple[MatLike, int]]
    ) -> Iterable[tuple[MatLike, HandLandmarkerResult | None, Gesture | None]]:
        for frame, timestamp in stream:
            detected_frame = self._hand_landmarker.detect(frame, timestamp)
            detection_result = self._hand_landmarker.latest_result
            analyzed = self._pose_analyzer.analyze(detection_result)
            classified = self._classifier.classify_gesture(analyzed)
            stabilized = self._stabilizer.update(classified)

            yield detected_frame, detection_result, stabilized

    def close(self) -> None:
        self._hand_landmarker.close()
