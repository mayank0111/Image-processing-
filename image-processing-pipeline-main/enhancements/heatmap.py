"""Movement heatmap generation.

Creates a 2D heatmap overlay showing where tracked subjects spend
the most time across the entire video.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


class HeatmapGenerator:
    """Generates movement heatmaps from accumulated tracking data."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        blur_sigma: int = 30,
        alpha: float = 0.6,
    ):
        """
        Args:
            resolution: (width, height) of the video.
            blur_sigma: Gaussian blur kernel size for smoothing.
            alpha: Heatmap overlay transparency (0=transparent, 1=opaque).
        """
        self.width, self.height = resolution
        self.blur_sigma = blur_sigma
        self.alpha = alpha
        self.accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def accumulate(self, centers: List[Tuple[float, float]]):
        """Add detection center points to the heatmap accumulator.

        Args:
            centers: List of (cx, cy) center coordinates.
        """
        for cx, cy in centers:
            x, y = int(cx), int(cy)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.accumulator[y, x] += 1

    def accumulate_from_results(self, track_positions: Dict[int, List]):
        """Accumulate from PipelineResults.track_positions format.

        Args:
            track_positions: Dict[track_id -> List[(frame_idx, cx, cy)]]
        """
        for tid, positions in track_positions.items():
            for frame_idx, cx, cy in positions:
                x, y = int(cx), int(cy)
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.accumulator[y, x] += 1

    def generate(self, background_frame: np.ndarray = None) -> np.ndarray:
        """Generate the final heatmap image.

        Args:
            background_frame: Optional frame to use as background.
                If None, uses a black background.

        Returns:
            Heatmap overlay as BGR numpy array.
        """
        # Apply Gaussian blur for smooth visualization
        kernel_size = self.blur_sigma * 2 + 1
        heatmap = cv2.GaussianBlur(
            self.accumulator, (kernel_size, kernel_size), 0
        )

        # Normalize to 0-255
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = np.zeros_like(heatmap, dtype=np.uint8)

        # Apply colormap (JET: blue=cold, red=hot)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Overlay on background
        if background_frame is not None:
            bg = background_frame.copy()
        else:
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Only overlay where there's actual data (threshold low values)
        mask = heatmap > 10
        result = bg.copy()
        result[mask] = cv2.addWeighted(
            bg[mask], 1 - self.alpha, heatmap_colored[mask], self.alpha, 0
        )

        return result

    def save(
        self,
        output_path: str,
        background_frame: np.ndarray = None,
        title: str = "Movement Heatmap",
    ):
        """Save heatmap as an image file.

        Args:
            output_path: Path to save the heatmap image.
            background_frame: Optional background frame.
            title: Title for the matplotlib figure.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        heatmap_img = self.generate(background_frame)

        # Save raw OpenCV image
        cv2.imwrite(output_path, heatmap_img)

        # Also save a matplotlib version with colorbar
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # Convert BGR to RGB for matplotlib
        ax.imshow(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")

        plt_path = output_path.replace(".png", "_matplotlib.png")
        fig.savefig(plt_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return output_path, plt_path
