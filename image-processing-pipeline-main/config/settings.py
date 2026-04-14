"""Centralized configuration for the tracking pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DetectorConfig:
    """Configuration for the object detector."""
    model_name: str = "yolo11m.pt"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    classes: List[int] = field(default_factory=lambda: [0])  # 0 = person in COCO
    device: str = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" fallback
    img_size: int = 1280


@dataclass
class TrackerConfig:
    """Configuration for the multi-object tracker."""
    tracker_type: str = "botsort"  # "botsort" or "bytetrack"
    reid_model: str = "osnet_x0_25_msmt17.pt"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30  # frames to keep lost tracks
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    cmc_method: str = "sof"  # camera motion compensation: ecc, orb, sof, sift
    frame_rate: int = 30


@dataclass
class AnnotatorConfig:
    """Configuration for frame annotation/visualization."""
    box_thickness: int = 2
    font_scale: float = 0.5
    show_confidence: bool = True
    show_trajectory: bool = True
    trail_length: int = 50  # frames of trajectory trail
    show_frame_counter: bool = True


@dataclass
class EnhancementConfig:
    """Configuration for optional enhancements."""
    enable_heatmap: bool = True
    enable_object_count: bool = True
    enable_model_comparison: bool = True
    heatmap_blur_sigma: int = 30
    heatmap_alpha: float = 0.6
    comparison_models: List[str] = field(
        default_factory=lambda: ["yolo11n.pt", "yolo11m.pt", "yolo11l.pt"]
    )


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    input_path: str = "data/input/cricket.mp4"
    output_dir: str = "data/output"
    screenshot_dir: str = "data/screenshots"
    screenshot_interval: int = 100  # save screenshot every N frames
    log_interval: int = 50  # print progress every N frames

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    annotator: AnnotatorConfig = field(default_factory=AnnotatorConfig)
    enhancements: EnhancementConfig = field(default_factory=EnhancementConfig)

    @property
    def output_video_path(self) -> str:
        input_stem = Path(self.input_path).stem
        tracker_name = self.tracker.tracker_type
        return str(Path(self.output_dir) / f"{input_stem}_{tracker_name}_tracked.mp4")

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent
