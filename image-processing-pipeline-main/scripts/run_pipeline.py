#!/usr/bin/env python3
"""CLI entry point for the multi-object tracking pipeline.

Usage:
    python scripts/run_pipeline.py --input data/input/cricket.mp4
    python scripts/run_pipeline.py --input video.mp4 --model yolo11l.pt --tracker bytetrack
    python scripts/run_pipeline.py --input video.mp4 --confidence 0.4 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PipelineConfig
from src.pipeline import TrackingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Object Detection & Tracking Pipeline"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--output-dir", "-o", default="data/output", help="Output directory"
    )
    parser.add_argument(
        "--model", "-m", default="yolo11m.pt",
        help="YOLO model name (yolo11n.pt, yolo11m.pt, yolo11l.pt)"
    )
    parser.add_argument(
        "--tracker", "-t", default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker algorithm"
    )
    parser.add_argument(
        "--confidence", "-c", type=float, default=0.25,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--device", "-d", default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Inference device"
    )
    parser.add_argument(
        "--no-trajectory", action="store_true",
        help="Disable trajectory trail visualization"
    )
    parser.add_argument(
        "--screenshot-interval", type=int, default=100,
        help="Save screenshot every N frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config from CLI args
    config = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        screenshot_interval=args.screenshot_interval,
    )
    config.detector.model_name = args.model
    config.detector.confidence = args.confidence
    config.detector.device = args.device
    config.tracker.tracker_type = args.tracker
    config.annotator.show_trajectory = not args.no_trajectory

    # Run pipeline
    pipeline = TrackingPipeline(config)
    results = pipeline.run()

    # Save results summary as JSON
    summary = {
        "total_frames": results.total_frames,
        "total_detections": results.total_detections,
        "total_unique_ids": results.total_unique_ids,
        "avg_detections_per_frame": round(results.avg_detections_per_frame, 2),
        "processing_fps": round(results.processing_fps, 2),
        "elapsed_seconds": round(results.elapsed_seconds, 2),
        "model": args.model,
        "tracker": args.tracker,
        "confidence": args.confidence,
        "output_video": config.output_video_path,
    }
    summary_path = Path(args.output_dir) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
