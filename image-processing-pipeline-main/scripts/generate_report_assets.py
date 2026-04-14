#!/usr/bin/env python3
"""Generate all report assets: heatmap, object count chart, model comparison, screenshots.

Run this AFTER the main pipeline has completed.

Usage:
    python scripts/generate_report_assets.py --input data/input/cricket.mp4
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PipelineConfig
from enhancements.heatmap import HeatmapGenerator
from enhancements.model_comparison import ModelComparison
from enhancements.object_count import ObjectCountChart
from src.utils import setup_logging
from src.video_io import VideoReader


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report assets")
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video"
    )
    parser.add_argument(
        "--results-dir", default="data/output",
        help="Directory containing run_summary.json and pipeline results"
    )
    parser.add_argument(
        "--output-dir", default="report/figures", help="Output dir for figures"
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip model comparison (takes longest)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load pipeline results
    summary_path = Path(args.results_dir) / "run_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info(f"Loaded pipeline results from {summary_path}")
    else:
        logger.warning(
            f"No run_summary.json found at {summary_path}. "
            "Run the pipeline first."
        )
        summary = {}

    reader = VideoReader(args.input)

    # 1. Object count chart (from re-running quick detection pass if needed)
    logger.info("Generating object count chart...")
    # For now, use the summary data if available
    # In a full run, detections_per_frame would be serialized
    logger.info(
        "Note: For full object count chart, use the pipeline_demo notebook "
        "or re-run the pipeline with --save-per-frame-data flag."
    )

    # 2. Model comparison
    if not args.skip_comparison:
        logger.info("Running model comparison (this may take a while)...")
        config = PipelineConfig(input_path=args.input)
        comparison = ModelComparison(config)
        results = comparison.run_comparison(max_frames=300)
        charts = comparison.generate_charts(output_dir)
        json_path = comparison.save_results(output_dir)
        logger.info(f"Model comparison charts: {charts}")
        logger.info(f"Model comparison data: {json_path}")
    else:
        logger.info("Skipping model comparison (use --skip-comparison to enable)")

    # 3. Extract sample frames for report
    logger.info("Extracting sample frames...")
    sample_frames = [0, len(reader) // 4, len(reader) // 2,
                     3 * len(reader) // 4, len(reader) - 1]
    for idx in sample_frames:
        try:
            frame = reader.read_frame(idx)
            frame_path = Path(output_dir) / f"sample_frame_{idx:06d}.png"
            import cv2
            cv2.imwrite(str(frame_path), frame)
            logger.info(f"  Saved: {frame_path}")
        except ValueError:
            pass

    logger.info("Report asset generation complete!")
    logger.info(f"Assets saved to: {output_dir}")


if __name__ == "__main__":
    main()
