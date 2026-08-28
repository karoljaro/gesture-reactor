from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
)


class FaceReactionPipeline:

    def handle_result(
        self,
        result: FaceLandmarkerResult,
        timestamp_ms: int
    ) -> None:
        if not result.face_landmarks:
            return

        print(f"Face detected: {timestamp_ms}")
