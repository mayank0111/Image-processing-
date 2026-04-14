"""Frame annotation and visualization module."""

import cv2
import numpy as np
import supervision as sv

from config.settings import AnnotatorConfig
from src.utils import frame_to_time


class Annotator:
    """Draws bounding boxes, ID labels, and trajectory trails on video frames.

    Uses the supervision library for production-quality annotations.
    """

    def __init__(self, config: AnnotatorConfig, fps: float = 30.0):
        self.config = config
        self.fps = fps

        # Supervision annotators
        self.box_annotator = sv.BoxAnnotator(
            thickness=config.box_thickness,
        )
        self.label_annotator = sv.LabelAnnotator(
            text_scale=config.font_scale,
            text_thickness=1,
            text_position=sv.Position.TOP_LEFT,
        )

        # Trajectory trail annotator
        if config.show_trajectory:
            self.trace_annotator = sv.TraceAnnotator(
                trace_length=config.trail_length,
                thickness=2,
                position=sv.Position.BOTTOM_CENTER,
            )
        else:
            self.trace_annotator = None

    def draw(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        frame_idx: int,
    ) -> np.ndarray:
        """Annotate a frame with tracked detections.

        Args:
            frame: Original BGR frame.
            detections: Tracked detections with tracker_id.
            frame_idx: Current frame index (for overlay).

        Returns:
            Annotated frame as numpy array.
        """
        annotated = frame.copy()

        if len(detections) == 0:
            return self._add_overlay(annotated, frame_idx, 0)

        # Build labels
        labels = self._build_labels(detections)

        # Draw trajectory trails first (behind boxes)
        if self.trace_annotator is not None:
            annotated = self.trace_annotator.annotate(
                scene=annotated, detections=detections
            )

        # Draw bounding boxes
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections
        )

        # Draw labels
        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Add frame info overlay
        annotated = self._add_overlay(annotated, frame_idx, len(detections))

        return annotated

    def _build_labels(self, detections: sv.Detections) -> list:
        """Build label strings for each detection."""
        labels = []
        for i in range(len(detections)):
            track_id = (
                detections.tracker_id[i]
                if detections.tracker_id is not None
                else "?"
            )
            if self.config.show_confidence and detections.confidence is not None:
                conf = detections.confidence[i]
                labels.append(f"ID:{track_id} ({conf:.2f})")
            else:
                labels.append(f"ID:{track_id}")
        return labels

    def _add_overlay(
        self, frame: np.ndarray, frame_idx: int, num_tracked: int
    ) -> np.ndarray:
        """Add frame counter and tracking info overlay."""
        if not self.config.show_frame_counter:
            return frame

        timestamp = frame_to_time(frame_idx, self.fps)
        text = f"Frame: {frame_idx} | Time: {timestamp} | Tracked: {num_tracked}"

        # Semi-transparent background bar
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(
            frame, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        return frame
