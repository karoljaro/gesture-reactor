from vision.async_landmark_processor import ResultCallback

from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarkerResult,
    FaceLandmarkerOptions
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


def create_face_landmarker_options(
    callback: ResultCallback[FaceLandmarkerResult]
) -> FaceLandmarkerOptions:
    return FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task",
        ),
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        num_faces=1,
        result_callback=callback,
    )
