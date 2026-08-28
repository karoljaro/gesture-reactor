from camera import Camera
from presentation.frame_display import FrameDisplay
from presentation.hand_landmark_renderer import HandLandmarkRenderer
from presentation.meme_display import MemeDisplay
from reaction.meme_reactor import MemeReactor
from vision.hand.pipeline.hand_reaction_pipeline import HandReactionPipeline
from vision.processor import VisionProcessor
from vision.pipelines.vision_detection_pipeline import VisionDetectionPipeline
from vision.face.pipeline.face_reaction_pipeline import FaceReactionPipeline


def main() -> None:
    camera = Camera()
    vision_processor = VisionProcessor()
    meme_display = MemeDisplay()
    frame_display = FrameDisplay()
    hand_landmark_renderer = HandLandmarkRenderer()

    meme_reactor = MemeReactor(
        on_meme=meme_display.show
    )

    hand_pipeline = HandReactionPipeline(
        on_gesture=meme_reactor.handle_gesture
    )

    face_pipeline = FaceReactionPipeline()

    vision_detection_pipeline = VisionDetectionPipeline(
        hand_pipeline.handle_result,
        face_pipeline.handle_result
    )

    try:
        camera_stream = camera.stream()
        processed_stream = vision_processor.process(camera_stream)
        vision_stream = vision_detection_pipeline.process(processed_stream)
        frame_display.show(vision_stream)

    except Exception as e:
        print(e)

    finally:
        camera.close()
        meme_display.close()
        frame_display.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
