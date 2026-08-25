from vision.async_landmark_processor import ResultCallback
from mediapipe.tasks.python.vision import FaceLandmarkerResult, FaceLandmarkerOptions
import mediapipe as mp


def create_face_landmarker_options(
    callback: ResultCallback[FaceLandmarkerResult]
) -> FaceLandmarkerOptions:
    return FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path="models/face_landmarker.task",
        ),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=callback,
    )
