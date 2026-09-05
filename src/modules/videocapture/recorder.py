import os
import cv2
from typing import Optional, Tuple

class DualVideoRecorder:
    """Encapsulate forward and bottom camera video recording."""

    def __init__(self, *,
                 timestr: str,
                 fps: float,
                 frame_size_forward: Tuple[int, int],
                 frame_size_bottom: Optional[Tuple[int, int]] = None,
                 output_dir: str = 'test_video_record',
                 fourcc_code: str = 'XVID') -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.forward_path = os.path.join(self.output_dir, f'auv_forward_{timestr}.avi')
        self.bottom_path = os.path.join(self.output_dir, f'auv_bottom_{timestr}.avi')

        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        self.forward_writer = cv2.VideoWriter(self.forward_path, fourcc, float(fps), frame_size_forward)
        self.bottom_writer = None
        if frame_size_bottom is not None:
            self.bottom_writer = cv2.VideoWriter(self.bottom_path, fourcc, float(fps), frame_size_bottom)

    def is_forward_open(self) -> bool:
        return self.forward_writer is not None and self.forward_writer.isOpened()

    def is_bottom_open(self) -> bool:
        return self.bottom_writer is not None and self.bottom_writer.isOpened()

    def write_forward(self, frame) -> None:
        if self.is_forward_open():
            self.forward_writer.write(frame)

    def write_bottom(self, frame) -> None:
        if self.is_bottom_open():
            self.bottom_writer.write(frame)

    def release(self) -> None:
        if self.is_forward_open():
            self.forward_writer.release()
        if self.is_bottom_open():
            self.bottom_writer.release()

    def paths(self) -> Tuple[str, Optional[str]]:
        return self.forward_path, (self.bottom_path if self.bottom_writer is not None else None)
