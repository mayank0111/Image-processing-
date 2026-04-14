"""Multi-object tracking module with BoT-SORT and ByteTrack support."""

import numpy as np
import supervision as sv

from config.settings import TrackerConfig


class Tracker:
    """Wraps BoT-SORT (via boxmot) or ByteTrack (via supervision) for
    multi-object tracking with persistent ID assignment.

    BoT-SORT provides:
    - ReID appearance embeddings for identity preservation
    - Camera motion compensation (GMC) for handling pans/zooms
    - Two-stage association (high + low confidence detections)

    ByteTrack provides:
    - Fast IoU-only association (no ReID)
    - Good baseline for comparison
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracker_type = config.tracker_type

        if config.tracker_type == "botsort":
            self._init_botsort(config)
        elif config.tracker_type == "bytetrack":
            self._init_bytetrack(config)
        else:
            raise ValueError(f"Unknown tracker type: {config.tracker_type}")

    def _init_botsort(self, config: TrackerConfig):
        """Initialize BoT-SORT tracker via boxmot."""
        from boxmot import BotSort

        self.tracker = BotSort(
            reid_weights=config.reid_model,
            device="cpu",  # ReID on CPU for MPS compatibility
            half=False,
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            proximity_thresh=config.proximity_thresh,
            appearance_thresh=config.appearance_thresh,
            cmc_method=config.cmc_method,
            frame_rate=config.frame_rate,
        )

    def _init_bytetrack(self, config: TrackerConfig):
        """Initialize ByteTrack tracker via supervision."""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_high_thresh,
            lost_track_buffer=config.track_buffer,
            minimum_matching_threshold=config.match_thresh,
            frame_rate=config.frame_rate,
        )

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Update tracker with new detections and return tracked objects.

        Args:
            detections: Detection results from the detector.
            frame: Current video frame (needed for BoT-SORT ReID extraction).

        Returns:
            sv.Detections with tracker_id field populated.
        """
        if self.tracker_type == "botsort":
            return self._update_botsort(detections, frame)
        else:
            return self._update_bytetrack(detections)

    def _update_botsort(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """Update BoT-SORT tracker.

        boxmot expects: numpy array of shape (N, 6) = [x1, y1, x2, y2, conf, cls]
        boxmot returns: numpy array of shape (M, 8) = [x1, y1, x2, y2, id, conf, cls, idx]
        """
        if len(detections) == 0:
            # Still update tracker to age existing tracks
            empty = np.empty((0, 6))
            self.tracker.update(empty, frame)
            return sv.Detections.empty()

        # Convert sv.Detections -> boxmot input format
        dets = np.column_stack([
            detections.xyxy,
            detections.confidence,
            detections.class_id,
        ])

        # Run tracker update
        tracked = self.tracker.update(dets, frame)

        if len(tracked) == 0:
            return sv.Detections.empty()

        # Convert boxmot output -> sv.Detections
        return sv.Detections(
            xyxy=tracked[:, 0:4].astype(np.float32),
            confidence=tracked[:, 5].astype(np.float32),
            class_id=tracked[:, 6].astype(int),
            tracker_id=tracked[:, 4].astype(int),
        )

    def _update_bytetrack(self, detections: sv.Detections) -> sv.Detections:
        """Update ByteTrack tracker."""
        if len(detections) == 0:
            return sv.Detections.empty()

        return self.tracker.update_with_detections(detections)

    def reset(self):
        """Reset tracker state (for new video or comparison runs)."""
        config = self.config
        if self.tracker_type == "botsort":
            self._init_botsort(config)
        else:
            self._init_bytetrack(config)
