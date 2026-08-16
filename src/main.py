from cv2.typing import MatLike
from collections.abc import Iterable

from camera import Camera
from vision.processor import VisionProcessor
from display import FrameDisplay
from vision.hand_landmarker import HandLandmarker
from vision.hand_pose_analyzer import HandPoseAnalyzer


def main() -> None:
    webcam = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    hand_landmarker = HandLandmarker()
    hand_pose_analyzer = HandPoseAnalyzer()

    try:
        camera_stream = webcam.execute()
        processed_stream = vision_processor.process(camera_stream)
        hand_landmarker_stream = hand_landmarker.detect(processed_stream)

        # landmarker_drawn_stream = (
        #     hand_landmarker.draw(frame, hand_landmarker.latest_result)
        #     for frame in hand_landmarker_stream
        # )

        def process_and_draw_stream() -> Iterable[MatLike]:
            for frame in hand_landmarker_stream:
                drawn_frame = hand_landmarker.draw(frame, hand_landmarker.latest_result)

                result = hand_pose_analyzer.analyze(hand_landmarker.latest_result)

                print(result)

                yield drawn_frame

        landmarker_drawn_stream = process_and_draw_stream()

        frame_display.show(landmarker_drawn_stream)

    finally:
        hand_landmarker.close()
        webcam.close()
        frame_display.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
