from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
)
from vision.face.face_expression_analyzer import FaceExpressionAnalyzer


class FaceReactionPipeline:
    def __init__(self) -> None:
        self._expression_analyzer = FaceExpressionAnalyzer()

    def handle_result(
        self,
        result: FaceLandmarkerResult,
        timestamp_ms: int
    ) -> None:
        print("-------------------------------------------------------------")
        print(f"Face detected: {timestamp_ms}")
        print(result.face_blendshapes)
        print("\n")
        analized = self._expression_analyzer.analyze(result)
        print(analized)
