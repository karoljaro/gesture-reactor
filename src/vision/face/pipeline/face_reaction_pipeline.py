from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
)
from vision.face.face_expression_analyzer import FaceExpressionAnalyzer
from vision.face.face_expression_classifier import FaceExpressionClassifier
from vision.classification_stabilizer import Stabilizer
from vision.face.types.expression import Expression
from collections.abc import Callable


class FaceReactionPipeline:
    def __init__(self, on_expression: Callable[[Expression, int], None]) -> None:
        self._expression_analyzer = FaceExpressionAnalyzer()
        self._classifier = FaceExpressionClassifier()
        self._stabilizer = Stabilizer[Expression]()
        self._on_expression = on_expression

    def handle_result(
        self,
        result: FaceLandmarkerResult,
        timestamp_ms: int
    ) -> None:
        analized = self._expression_analyzer.analyze(result)
        classified = self._classifier.classify(analized)

        stabilized = self._stabilizer.stabilize(
            classified,
            lambda expression: expression is Expression.NEUTRAL,
        )

        if stabilized is not None:
            self._on_expression(stabilized, timestamp_ms)
