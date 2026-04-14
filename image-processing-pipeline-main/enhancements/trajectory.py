"""Trajectory trail visualization enhancement.

Draws movement trails behind each tracked subject, showing their path
over the last N frames. Color-coded by track ID for visual clarity.
"""

from collections import defaultdict, deque
from typing import Dict, Tuple

import cv2
import numpy as np
import supervision as sv

from src.utils import get_color_for_id


class TrajectoryVisualizer:
    """Maintains and draws trajectory trails for tracked objects.

    This is an alternative to supervision's built-in TraceAnnotator,
    providing more control over rendering (e.g., fading trails, custom colors).
    """

    def __init__(self, trail_length: int = 50, line_thickness: int = 2):
        self.trail_length = trail_length
        self.line_thickness = line_thickness
        # track_id -> deque of (cx, cy) center points
        self.trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def update(self, detections: sv.Detections):
        """Add current detection centers to trail history."""
        if detections.tracker_id is None or len(detections) == 0:
            return

        for i in range(len(detections)):
            tid = int(detections.tracker_id[i])
            x1, y1, x2, y2 = detections.xyxy[i]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            self.trails[tid].append((cx, cy))

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw trajectory trails on the frame.

        Trails fade from full opacity (current position) to transparent
        (oldest position) for a smooth visual effect.
        """
        overlay = frame.copy()

        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue

            color = get_color_for_id(tid)
            points = list(trail)

            for j in range(1, len(points)):
                # Fade: more recent = more opaque
                alpha = j / len(points)
                thickness = max(1, int(self.line_thickness * alpha))

                pt1 = points[j - 1]
                pt2 = points[j]
                cv2.line(overlay, pt1, pt2, color, thickness)

        # Blend overlay with original for slight transparency
        return cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    def get_all_positions(self) -> Dict[int, list]:
        """Return all tracked positions (for heatmap generation)."""
        return {tid: list(trail) for tid, trail in self.trails.items()}

    def reset(self):
        """Clear all trajectory data."""
        self.trails.clear()
