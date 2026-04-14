#!/usr/bin/env python3
"""Download a public cricket/sports video for the tracking pipeline.

Usage:
    python scripts/download_video.py
    python scripts/download_video.py --url "https://youtube.com/watch?v=..." --duration 60
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Default: a publicly available cricket highlights video
# Replace this URL with your chosen public video
DEFAULT_URL = "https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_VIDEO_ID"
DEFAULT_DURATION = 60  # seconds


def parse_args():
    parser = argparse.ArgumentParser(description="Download public video for tracking")
    parser.add_argument(
        "--url", default=DEFAULT_URL, help="YouTube or public video URL"
    )
    parser.add_argument(
        "--duration", type=int, default=DEFAULT_DURATION,
        help="Duration in seconds to trim (0 = full video)"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="Start time in seconds"
    )
    parser.add_argument(
        "--output", default="data/input/cricket.mp4", help="Output file path"
    )
    parser.add_argument(
        "--resolution", default="720", help="Max video height (e.g., 720, 1080)"
    )
    return parser.parse_args()


def download_video(url: str, output_path: str, resolution: str):
    """Download video using yt-dlp."""
    print(f"Downloading video from: {url}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.replace(".mp4", "_full.mp4")

    cmd = [
        "yt-dlp",
        "-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "--merge-output-format", "mp4",
        "-o", temp_path,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"yt-dlp error: {result.stderr}")
        sys.exit(1)

    print(f"Downloaded to: {temp_path}")
    return temp_path


def trim_video(input_path: str, output_path: str, start: int, duration: int):
    """Trim video using ffmpeg."""
    if duration <= 0:
        # No trimming, just rename
        Path(input_path).rename(output_path)
        return

    print(f"Trimming: start={start}s, duration={duration}s")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")
        sys.exit(1)

    # Clean up full video
    Path(input_path).unlink(missing_ok=True)
    print(f"Trimmed video saved to: {output_path}")


def save_metadata(url: str, output_path: str, duration: int, start: int):
    """Save video source metadata for attribution."""
    metadata = {
        "source_url": url,
        "start_seconds": start,
        "duration_seconds": duration,
        "output_path": output_path,
        "note": "Publicly available video used for academic/evaluation purposes only.",
    }
    metadata_path = Path(output_path).parent / "video_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")


def main():
    args = parse_args()

    if "REPLACE_WITH" in args.url:
        print("=" * 60)
        print("Please provide a valid public video URL.")
        print("Usage: python scripts/download_video.py --url 'YOUR_URL'")
        print()
        print("Recommended: search YouTube for 'cricket match highlights'")
        print("or 'IPL highlights 2024' and use the URL of a suitable clip.")
        print("=" * 60)
        sys.exit(1)

    # Download
    temp_path = download_video(args.url, args.output, args.resolution)

    # Trim
    trim_video(temp_path, args.output, args.start, args.duration)

    # Save metadata
    save_metadata(args.url, args.output, args.duration, args.start)

    print(f"\nVideo ready at: {args.output}")
    print("Run the pipeline with:")
    print(f"  python scripts/run_pipeline.py --input {args.output}")


if __name__ == "__main__":
    main()
