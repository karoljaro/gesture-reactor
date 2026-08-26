from queue import Queue, Empty, Full
from typing import Protocol
from collections.abc import Callable
import mediapipe as mp
from mediapipe.tasks.python.vision.core.image_processing_options import ImageProcessingOptions

type ResultCallback[TResult] = Callable[[TResult, mp.Image, int], None]

type LandmarkerType[TOptions] = type[LandmarkerFactory[TOptions]]


class LandmarkedInstance(Protocol):
    def detect_async(
        self,
        image: mp.Image,
        timestamp_ms: int,
        image_processing_options: ImageProcessingOptions | None = None,
    ) -> None: ...

    def close(self) -> None: ...


class LandmarkerFactory[TOptions](Protocol):
    @classmethod
    def create_from_options(cls, options: TOptions) -> LandmarkedInstance: ...


class LandmarkerOptionsFactory[TOptions, TResult](Protocol):
    def __call__(
        self,
        callback: ResultCallback[TResult],
    ) -> TOptions: ...


class AsyncLandmarkProcessor[
    TResult,
    TLandmarkerOptions,
]:
    def __init__(
        self,
        landmarker: LandmarkerType[TLandmarkerOptions],
        options_factory: LandmarkerOptionsFactory[TLandmarkerOptions, TResult],
    ) -> None:
        self._results: Queue[tuple[TResult, int]] = Queue(maxsize=1)
        options = options_factory(self._on_result)

        self._landmarker = landmarker.create_from_options(options)

    def submit(self, mp_image: mp.Image, timestamp_ms: int) -> None:
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def try_get_result(self) -> tuple[TResult, int] | None:
        try:
            result, result_timestamp = self._results.get_nowait()
        except Empty:
            return None

        return result, result_timestamp

    def close(self) -> None:
        self._landmarker.close()

    def _on_result(self, result: TResult, _output_image: mp.Image, timestamp_ms: int) -> None:
        packet = result, timestamp_ms

        try:
            self._results.put_nowait(packet)
        except Full:
            try:
                self._results.get_nowait()
            except Empty:
                pass

            self._results.put_nowait(packet)
