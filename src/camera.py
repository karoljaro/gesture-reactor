import cv2
from cv2.typing import MatLike


class Camera:
    def __init__(self, device_index: int = 0, window_name: str = "Webcam Feed"):
        self._device_index = device_index
        self._window_name = window_name
        self._capture = cv2.VideoCapture(device_index)

    def execute(self) -> None:
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open camera with device index {self._device_index}")

        print("Webcam access successfully.")

        cv2.namedWindow(self._window_name)

        try:
            while True:
                frame = self._read_frame()
                cv2.imshow(self._window_name, frame)

                if self._should_close():
                    break

        finally:
            self.close()

    def _read_frame(self) -> MatLike:
        success, frame = self._capture.read()
        if not success:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def _should_close(self) -> bool:
        try:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True

            return cv2.getWindowProperty(
                self._window_name,
                cv2.WND_PROP_VISIBLE,
            ) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        self._capture.release()
        cv2.destroyAllWindows()
