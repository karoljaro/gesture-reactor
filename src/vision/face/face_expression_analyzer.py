from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.components.containers.category import Category
from functools import cache
import re
from typing import cast


@cache
def _to_snake_case(name: str) -> str:
    if name.startswith("_"):
        return name

    return re.sub(
        r"(?<!^)(?=[A-Z])",
        "_",
        name,
    ).lower()


class FaceExpressionAnalyzer:
    def analyze(
        self,
        latest_result: FaceLandmarkerResult | None,
    ) -> dict[str, float] | None:
        if latest_result is None or not latest_result.face_blendshapes:
            return None

        return {
            _to_snake_case(category.category_name): category.score
            for category in cast(list[Category], latest_result.face_blendshapes[0])
            if category.category_name is not None
        }
