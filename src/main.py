from camera import Camera
from vision.processor import VisionProcessor
from display import FrameDisplay


def main() -> None:
    camera = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()

    camera_stream = camera.execute()
    processed_stream = vision_processor.process(camera_stream)
    frame_display.show(processed_stream)


if __name__ == "__main__":
    main()
