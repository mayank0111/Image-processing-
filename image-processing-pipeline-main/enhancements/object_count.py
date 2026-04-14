"""Object count over time visualization.

Generates a time-series chart showing the number of tracked subjects
per frame/second throughout the video.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


class ObjectCountChart:
    """Generates object count over time charts from pipeline results."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def generate(
        self,
        detections_per_frame: List[int],
        output_path: str,
        title: str = "Tracked Subjects Over Time",
    ) -> str:
        """Generate and save the object count chart.

        Args:
            detections_per_frame: List of detection counts, one per frame.
            output_path: Path to save the chart image.
            title: Chart title.

        Returns:
            Path to the saved chart.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        frames = np.arange(len(detections_per_frame))
        times = frames / self.fps

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Raw per-frame count
        ax1.plot(times, detections_per_frame, alpha=0.4, linewidth=0.5,
                 color="steelblue", label="Per-frame")

        # Smoothed (rolling average over 1 second)
        window = max(1, int(self.fps))
        smoothed = np.convolve(
            detections_per_frame, np.ones(window) / window, mode="same"
        )
        ax1.plot(times, smoothed, linewidth=2, color="darkblue",
                 label=f"Rolling avg ({window} frames)")

        ax1.set_ylabel("Number of Tracked Subjects", fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.legend(loc="upper right")
        ax1.set_ylim(bottom=0)

        # Distribution histogram
        ax2.hist(detections_per_frame, bins=range(
            max(detections_per_frame) + 2
        ), color="steelblue", edgecolor="white", alpha=0.8)
        ax2.set_xlabel("Time (seconds)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Detection Count Distribution", fontsize=12)

        # Fix x-axis label for histogram
        ax2.set_xlabel("Number of Tracked Subjects per Frame")

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_summary_stats(self, detections_per_frame: List[int]) -> dict:
        """Compute summary statistics for the detection counts."""
        counts = np.array(detections_per_frame)
        return {
            "min_tracked": int(counts.min()),
            "max_tracked": int(counts.max()),
            "mean_tracked": round(float(counts.mean()), 2),
            "median_tracked": round(float(np.median(counts)), 2),
            "std_tracked": round(float(counts.std()), 2),
            "frames_with_zero": int(np.sum(counts == 0)),
            "total_frames": len(counts),
        }
