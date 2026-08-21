from camera import Camera
from presentation.frame_display import FrameDisplay
from presentation.hand_landmark_renderer import HandLandmarkRenderer
from presentation.meme_display import MemeDisplay
from reaction.meme_reactor import MemeReactor
from vision.hand.pipeline.hand_reaction_pipeline import HandReactionPipeline
from vision.processor import VisionProcessor


def main() -> None:
    webcam = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    hand_landmark_renderer = HandLandmarkRenderer()
    meme_reactor = MemeReactor()
    meme_display = MemeDisplay()
    hand_reaction_pipeline = HandReactionPipeline()

    try:
        camera_stream = webcam.execute()
        processed_stream = vision_processor.process(camera_stream)
        hand_reaction_stream = hand_reaction_pipeline.process(processed_stream)
        rendered_hand_stream = hand_landmark_renderer.render(hand_reaction_stream)
        meme_reaction_stream = meme_reactor.react(rendered_hand_stream)
        meme_display_stream = meme_display.show(meme_reaction_stream)
        frame_display.show(meme_display_stream)

    finally:
        hand_reaction_pipeline.close()
        webcam.close()
        meme_display.close()
        frame_display.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
