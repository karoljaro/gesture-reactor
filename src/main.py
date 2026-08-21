from collections.abc import Iterable

from cv2.typing import MatLike

from camera import Camera
from display import FrameDisplay
from meme_display import MemeDisplay
from reaction.meme_reactor import MemeReactor
from vision.hand.pipeline.hand_reaction_pipeline import HandReactionPipeline
from vision.processor import VisionProcessor
from vision.types import Gesture


def process_reactions(
    stream: Iterable[tuple[MatLike, Gesture | None]],
    meme_reactor: MemeReactor,
    meme_display: MemeDisplay,
) -> Iterable[MatLike]:
    for frame, stabilized_gesture in stream:
        if stabilized_gesture is not None:
            print(f"Stabilized gesture: {stabilized_gesture}")
            meme_path = meme_reactor.react(stabilized_gesture)
            meme_display.show(meme_path)
            print(f"Selected meme path: {meme_path}")

        yield frame


def main() -> None:
    webcam = Camera()
    vision_processor = VisionProcessor()
    frame_display = FrameDisplay()
    meme_reactor = MemeReactor()
    meme_display = MemeDisplay()
    hand_reaction_pipeline = HandReactionPipeline()

    try:
        camera_stream = webcam.execute()
        processed_stream = vision_processor.process(camera_stream)
        hand_reaction_stream = hand_reaction_pipeline.process(processed_stream)
        reaction_stream = process_reactions(
            hand_reaction_stream,
            meme_reactor,
            meme_display,
        )
        frame_display.show(reaction_stream)

    finally:
        hand_reaction_pipeline.close()
        webcam.close()
        frame_display.close()
        print("Resources released. Exiting the program.")


if __name__ == "__main__":
    main()
