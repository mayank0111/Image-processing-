"""Utility functions for the tracking pipeline."""

import hashlib
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a deterministic, visually distinct BGR color for a track ID."""
    hash_bytes = hashlib.md5(str(track_id).encode()).digest()
    # Use first 3 bytes, ensure brightness > 100 for visibility
    r = 100 + hash_bytes[0] % 156
    g = 100 + hash_bytes[1] % 156
    b = 100 + hash_bytes[2] % 156
    return (b, g, r)  # BGR for OpenCV


def frame_to_time(frame_idx: int, fps: float) -> str:
    """Convert frame index to MM:SS.ms timestamp."""
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def save_screenshot(frame: np.ndarray, frame_idx: int, output_dir: str):
    """Save a frame as a screenshot PNG."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(filepath), frame)
    return str(filepath)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure pipeline logging."""
    logger = logging.getLogger("tracking_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
