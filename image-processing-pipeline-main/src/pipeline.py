"""Main pipeline orchestrator: detect -> track -> annotate -> write."""

import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from config.settings import PipelineConfig
from src.annotator import Annotator
from src.detector import Detector
from src.tracker import Tracker
from src.utils import save_screenshot, setup_logging
from src.video_io import VideoReader, VideoWriter


@dataclass
class PipelineResults:
    """Aggregated results from a pipeline run."""
    total_frames: int = 0
    total_detections: int = 0
    total_unique_ids: int = 0
    avg_detections_per_frame: float = 0.0
    processing_fps: float = 0.0
    elapsed_seconds: float = 0.0
    detections_per_frame: List[int] = field(default_factory=list)
    unique_ids_seen: set = field(default_factory=set)
    track_positions: Dict[int, List] = field(default_factory=dict)

    def finalize(self):
        """Compute final aggregate metrics."""
        self.total_unique_ids = len(self.unique_ids_seen)
        if self.total_frames > 0:
            self.avg_detections_per_frame = self.total_detections / self.total_frames
            self.processing_fps = self.total_frames / max(self.elapsed_seconds, 1e-6)


class TrackingPipeline:
    """Orchestrates the full detection -> tracking -> annotation pipeline.

    Usage:
        config = PipelineConfig(input_path="video.mp4")
        pipeline = TrackingPipeline(config)
        results = pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging()

        # Initialize components
        self.video_reader = VideoReader(config.input_path)
        self.detector = Detector(config.detector)
        self.tracker = Tracker(config.tracker)
        self.annotator = Annotator(
            config.annotator, fps=self.video_reader.fps
        )

        # Update tracker frame rate from video
        config.tracker.frame_rate = int(self.video_reader.fps)

        self.logger.info(f"Input: {config.input_path}")
        self.logger.info(
            f"Video: {self.video_reader.width}x{self.video_reader.height} "
            f"@ {self.video_reader.fps:.1f} FPS, "
            f"{self.video_reader.total_frames} frames"
        )
        self.logger.info(f"Detector: {config.detector.model_name} on {self.detector.device}")
        self.logger.info(f"Tracker: {config.tracker.tracker_type}")

    def run(self) -> PipelineResults:
        """Run the full pipeline on the input video.

        Returns:
            PipelineResults with aggregated metrics and per-frame data.
        """
        results = PipelineResults()
        video_writer = VideoWriter(
            self.config.output_video_path,
            fps=self.video_reader.fps,
            resolution=self.video_reader.resolution,
        )

        self.logger.info(f"Output: {self.config.output_video_path}")
        self.logger.info("Starting pipeline...")

        start_time = time.time()

        for frame_idx, frame in enumerate(
            tqdm(self.video_reader, total=len(self.video_reader), desc="Processing")
        ):
            # Detect
            detections = self.detector.detect(frame)

            # Track
            tracked = self.tracker.update(detections, frame)

            # Annotate
            annotated = self.annotator.draw(frame, tracked, frame_idx)

            # Write output frame
            video_writer.write(annotated)

            # Accumulate statistics
            self._accumulate_stats(results, tracked, frame_idx)

            # Save screenshots at intervals
            if (
                frame_idx % self.config.screenshot_interval == 0
                and frame_idx > 0
            ):
                path = save_screenshot(
                    annotated, frame_idx, self.config.screenshot_dir
                )
                self.logger.info(f"Screenshot saved: {path}")

        # Finalize
        results.elapsed_seconds = time.time() - start_time
        results.total_frames = len(self.video_reader)
        results.finalize()
        video_writer.release()

        self._log_summary(results)
        return results

    def _accumulate_stats(
        self, results: PipelineResults, tracked, frame_idx: int
    ):
        """Accumulate per-frame tracking statistics."""
        num_tracked = len(tracked)
        results.total_detections += num_tracked
        results.detections_per_frame.append(num_tracked)

        if tracked.tracker_id is not None:
            for i in range(num_tracked):
                tid = int(tracked.tracker_id[i])
                results.unique_ids_seen.add(tid)

                # Store center positions for heatmap/trajectory
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if tid not in results.track_positions:
                    results.track_positions[tid] = []
                results.track_positions[tid].append(
                    (frame_idx, float(cx), float(cy))
                )

    def _log_summary(self, results: PipelineResults):
        """Log pipeline run summary."""
        self.logger.info("=" * 60)
        self.logger.info("Pipeline Complete")
        self.logger.info(f"  Frames processed:  {results.total_frames}")
        self.logger.info(f"  Processing FPS:    {results.processing_fps:.1f}")
        self.logger.info(f"  Elapsed time:      {results.elapsed_seconds:.1f}s")
        self.logger.info(f"  Total detections:  {results.total_detections}")
        self.logger.info(f"  Unique IDs:        {results.total_unique_ids}")
        self.logger.info(f"  Avg tracked/frame: {results.avg_detections_per_frame:.1f}")
        self.logger.info(f"  Output video:      {self.config.output_video_path}")
        self.logger.info("=" * 60)
