from camera import Camera
from vision.processor import VisionProcessor
from display import FrameDisplay
from vision.hand_landmarker import HandLandmarker


def main() -> None:
    camera = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    hand_landmarker = HandLandmarker()

    camera_stream = camera.execute()
    processed_stream = vision_processor.process(camera_stream)
    hand_landmarker_stream = hand_landmarker.detect(processed_stream)
    frame_display.show(hand_landmarker_stream)


if __name__ == "__main__":
    main()
