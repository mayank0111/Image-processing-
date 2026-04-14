"""Video reading and writing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple


class VideoReader:
    """Generator-based video frame reader."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def __iter__(self) -> Iterator[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def __len__(self) -> int:
        return self.total_frames

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a specific frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx}")
        return frame

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()


class VideoWriter:
    """Writes annotated frames to an output video file."""

    def __init__(self, path: str, fps: float, resolution: Tuple[int, int]):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, resolution)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer for: {path}")

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.release()
