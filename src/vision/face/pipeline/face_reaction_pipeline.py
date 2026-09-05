from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
)
from vision.face.face_expression_analyzer import FaceExpressionAnalyzer
from vision.face.face_expression_classifier import FaceExpressionClassifier
from vision.classification_stabilizer import Stabilizer
from vision.face.types.expression import Expression


class FaceReactionPipeline:
    def __init__(self) -> None:
        self._expression_analyzer = FaceExpressionAnalyzer()
        self._classifier = FaceExpressionClassifier()
        self._stabilizer = Stabilizer[Expression]()

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
            print(stabilized)
