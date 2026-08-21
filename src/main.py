from cv2.typing import MatLike
from collections.abc import Iterable

from camera import Camera
from vision.processor import VisionProcessor
from display import FrameDisplay
from vision.hand.hand_landmarker import HandLandmarker
from vision.hand.hand_pose_analyzer import HandPoseAnalyzer
from vision.hand.hand_gesture_classifier import HandGestureClassifier
from vision.hand.hand_gesture_stabilizer import HandGestureStabilizer
from reaction.meme_reactor import MemeReactor
from meme_display import MemeDisplay


def main() -> None:
    webcam = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    hand_landmarker = HandLandmarker()
    hand_pose_analyzer = HandPoseAnalyzer()
    gesture_classifier = HandGestureClassifier()
    gesture_stabilizer = HandGestureStabilizer()
    meme_reactor = MemeReactor()
    meme_display = MemeDisplay()

    try:
        camera_stream = webcam.execute()
        processed_stream = vision_processor.process(camera_stream)
        hand_landmarker_stream = hand_landmarker.detect(processed_stream)

        def process_and_draw_stream() -> Iterable[MatLike]:
            for frame in hand_landmarker_stream:
                drawn_frame = hand_landmarker.draw(frame, hand_landmarker.latest_result)

                result = hand_pose_analyzer.analyze(hand_landmarker.latest_result)

                gesture = gesture_classifier.classify_gesture(result)

                stabilized_gesture = gesture_stabilizer.update(gesture)

                if stabilized_gesture is not None:
                    print(f"Stabilized gesture: {stabilized_gesture}")
                    meme_path = meme_reactor.react(stabilized_gesture)
                    meme_display.show(meme_path)
                    print(f"Selected meme path: {meme_path}")

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
