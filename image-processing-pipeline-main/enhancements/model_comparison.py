"""Model comparison: run multiple detector/tracker combinations and compare.

Compares YOLOv11 variants (nano, medium, large) and tracker algorithms
(BoT-SORT vs ByteTrack) on the same video.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from config.settings import PipelineConfig
from src.detector import Detector
from src.tracker import Tracker
from src.utils import setup_logging
from src.video_io import VideoReader


@dataclass
class ComparisonResult:
    """Metrics for a single detector+tracker run."""
    model_name: str
    tracker_type: str
    total_frames: int
    processing_fps: float
    total_detections: int
    total_unique_ids: int
    avg_detections_per_frame: float


class ModelComparison:
    """Runs the same video through multiple model/tracker combinations."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging()
        self.results: List[ComparisonResult] = []

    def run_comparison(
        self,
        models: List[str] = None,
        trackers: List[str] = None,
        max_frames: int = 300,
    ) -> List[ComparisonResult]:
        """Run comparison across model/tracker combinations.

        Args:
            models: List of YOLO model names to compare.
            trackers: List of tracker types to compare.
            max_frames: Max frames to process per run (for speed).

        Returns:
            List of ComparisonResult for each combination.
        """
        if models is None:
            models = ["yolo11n.pt", "yolo11m.pt", "yolo11l.pt"]
        if trackers is None:
            trackers = ["botsort", "bytetrack"]

        self.results = []

        for model_name in models:
            for tracker_type in trackers:
                self.logger.info(
                    f"Running: {model_name} + {tracker_type}"
                )
                result = self._run_single(
                    model_name, tracker_type, max_frames
                )
                self.results.append(result)
                self.logger.info(
                    f"  FPS: {result.processing_fps:.1f}, "
                    f"IDs: {result.total_unique_ids}, "
                    f"Avg det/frame: {result.avg_detections_per_frame:.1f}"
                )

        return self.results

    def _run_single(
        self, model_name: str, tracker_type: str, max_frames: int
    ) -> ComparisonResult:
        """Run a single detector+tracker combination."""
        # Configure
        det_config = self.config.detector
        det_config.model_name = model_name
        trk_config = self.config.tracker
        trk_config.tracker_type = tracker_type

        detector = Detector(det_config)
        tracker = Tracker(trk_config)
        reader = VideoReader(self.config.input_path)

        total_detections = 0
        unique_ids = set()
        frame_count = 0
        start_time = time.time()

        for frame in tqdm(
            reader, total=min(len(reader), max_frames),
            desc=f"{model_name}+{tracker_type}", leave=False
        ):
            if frame_count >= max_frames:
                break

            detections = detector.detect(frame)
            tracked = tracker.update(detections, frame)

            total_detections += len(tracked)
            if tracked.tracker_id is not None:
                for tid in tracked.tracker_id:
                    unique_ids.add(int(tid))

            frame_count += 1

        elapsed = time.time() - start_time

        return ComparisonResult(
            model_name=model_name,
            tracker_type=tracker_type,
            total_frames=frame_count,
            processing_fps=frame_count / max(elapsed, 1e-6),
            total_detections=total_detections,
            total_unique_ids=len(unique_ids),
            avg_detections_per_frame=(
                total_detections / max(frame_count, 1)
            ),
        )

    def generate_charts(self, output_dir: str) -> List[str]:
        """Generate comparison charts and save to disk.

        Returns:
            List of paths to generated chart images.
        """
        if not self.results:
            return []

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        paths = []

        # Prepare data
        labels = [
            f"{r.model_name.replace('.pt','')}\n{r.tracker_type}"
            for r in self.results
        ]
        fps_values = [r.processing_fps for r in self.results]
        id_values = [r.total_unique_ids for r in self.results]
        det_values = [r.avg_detections_per_frame for r in self.results]

        # Chart 1: Processing speed (FPS)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = sns.color_palette("Blues_d", len(self.results))
        axes[0].barh(labels, fps_values, color=colors)
        axes[0].set_xlabel("Frames Per Second")
        axes[0].set_title("Processing Speed")
        for i, v in enumerate(fps_values):
            axes[0].text(v + 0.5, i, f"{v:.1f}", va="center")

        # Chart 2: Unique IDs (fewer = better tracking consistency)
        colors2 = sns.color_palette("Oranges_d", len(self.results))
        axes[1].barh(labels, id_values, color=colors2)
        axes[1].set_xlabel("Total Unique IDs Assigned")
        axes[1].set_title("ID Consistency (fewer = better)")
        for i, v in enumerate(id_values):
            axes[1].text(v + 0.5, i, str(v), va="center")

        # Chart 3: Average detections per frame
        colors3 = sns.color_palette("Greens_d", len(self.results))
        axes[2].barh(labels, det_values, color=colors3)
        axes[2].set_xlabel("Avg Detections / Frame")
        axes[2].set_title("Detection Rate")
        for i, v in enumerate(det_values):
            axes[2].text(v + 0.1, i, f"{v:.1f}", va="center")

        plt.suptitle("Model & Tracker Comparison", fontsize=16, y=1.02)
        plt.tight_layout()
        chart_path = str(Path(output_dir) / "model_comparison.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(chart_path)

        return paths

    def save_results(self, output_dir: str) -> str:
        """Save comparison results as JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        data = [
            {
                "model": r.model_name,
                "tracker": r.tracker_type,
                "fps": round(r.processing_fps, 2),
                "unique_ids": r.total_unique_ids,
                "avg_detections_per_frame": round(r.avg_detections_per_frame, 2),
                "total_frames": r.total_frames,
            }
            for r in self.results
        ]
        path = str(Path(output_dir) / "model_comparison.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
