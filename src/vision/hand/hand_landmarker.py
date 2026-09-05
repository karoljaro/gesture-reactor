from vision.async_landmark_processor import ResultCallback

from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


def create_hand_landmarker_options(
    callback: ResultCallback[HandLandmarkerResult],
) -> HandLandmarkerOptions:
    return HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/hand_landmarker.task",
        ),
        running_mode=VisionTaskRunningMode.LIVE_STREAM,
        result_callback=callback,
    )
