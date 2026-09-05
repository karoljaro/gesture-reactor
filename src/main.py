from camera import Camera
from presentation.frame_display import FrameDisplay
from presentation.hand_landmark_renderer import HandLandmarkRenderer
from presentation.meme_display import MemeDisplay
from reaction.meme_reactor import MemeReactor
from vision.hand.pipeline.hand_reaction_pipeline import HandReactionPipeline
from vision.processor import VisionProcessor
from vision.pipelines.vision_detection_pipeline import VisionDetectionPipeline
from vision.face.pipeline.face_reaction_pipeline import FaceReactionPipeline
from vision.pipelines.debug_render_pipeline import DebugRenderPipeline
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult


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

    face_pipeline = FaceReactionPipeline(
        on_expression=meme_reactor.handle_expression
    )

    debug_render_pipelne = DebugRenderPipeline[
        HandLandmarkerResult
    ](
        on_hand_draw=hand_landmark_renderer.draw
    )

    vision_detection_pipeline = VisionDetectionPipeline(
        hand_pipeline.handle_result,
        face_pipeline.handle_result,
        on_hand_debug=debug_render_pipelne.handle_hand_result
    )

    try:
        camera_stream = camera.stream()
        processed_stream = vision_processor.process(camera_stream)
        vision_stream = vision_detection_pipeline.process(processed_stream)
        debug_stream = debug_render_pipelne.process(vision_stream)
        frame_display.show(debug_stream)

    except Exception as e:
        print(e)

    finally:
        camera.close()
        meme_display.close()
        frame_display.close()
        vision_detection_pipeline.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
