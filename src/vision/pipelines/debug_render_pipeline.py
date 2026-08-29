from collections.abc import Iterable, Iterator, Callable
from cv2.typing import MatLike

type DrawHandLandmarkerHandler[TResult] = Callable[[MatLike, TResult], MatLike | None]


class DebugRenderPipeline[TResult]:
    def __init__(self, on_hand_draw: DrawHandLandmarkerHandler[TResult]) -> None:
        self._on_hand_draw = on_hand_draw
        self._latest_hand_result: TResult | None = None

    def handle_hand_result(
        self,
        result: TResult,
        _timestamp_ms: int
    ) -> None:
        self._latest_hand_result = result

    def process(self, stream: Iterable[MatLike]) -> Iterator[MatLike]:
        for frame in stream:
            if self._latest_hand_result is None:
                yield frame
                continue

            drawn_frame = self._on_hand_draw(
                frame,
                self._latest_hand_result
            )

            yield drawn_frame if drawn_frame is not None else frame
