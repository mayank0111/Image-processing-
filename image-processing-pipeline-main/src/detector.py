"""Object detection module using YOLOv11."""

import numpy as np
import supervision as sv
from ultralytics import YOLO

from config.settings import DetectorConfig


class Detector:
    """Wraps ultralytics YOLO model for person detection.

    Returns supervision.Detections objects for seamless integration
    with the tracker and annotator modules.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = YOLO(config.model_name)

        # Attempt to use configured device, fall back to CPU
        try:
            self.model.to(config.device)
            self.device = config.device
        except Exception:
            self.model.to("cpu")
            self.device = "cpu"

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            sv.Detections with xyxy, confidence, and class_id arrays.
        """
        results = self.model(
            frame,
            conf=self.config.confidence,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            imgsz=self.config.img_size,
            verbose=False,
        )[0]

        return sv.Detections.from_ultralytics(results)

    @property
    def model_name(self) -> str:
        return self.config.model_name
