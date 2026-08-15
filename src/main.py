from camera import Camera
from vision.processor import VisionProcessor
from display import FrameDisplay
from vision.hand_landmarker import HandLandmarker


def main() -> None:
    webcam = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    hand_landmarker = HandLandmarker()

    try:
        camera_stream = webcam.execute()
        processed_stream = vision_processor.process(camera_stream)
        hand_landmarker_stream = hand_landmarker.detect(processed_stream)

        landmarker_drawn_stream = (
            hand_landmarker.draw(frame, hand_landmarker.latest_result)
            for frame in hand_landmarker_stream
        )

        frame_display.show(landmarker_drawn_stream)

    finally:
        hand_landmarker.close()
        webcam.close()
        frame_display.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
