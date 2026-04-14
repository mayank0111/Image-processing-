# Multi-Object Detection & Persistent ID Tracking Pipeline

A computer vision pipeline for detecting and tracking multiple subjects in sports video footage, with persistent unique ID assignment that handles occlusion, camera motion, scale changes, and similar-looking subjects.

## Results Preview

| Annotated Tracking Output | Movement Heatmap |
|---|---|
| Bounding boxes + IDs + trajectory trails | Spatial density of player positions |

*See `data/screenshots/` and `report/figures/` for full results after running the pipeline.*

## Architecture

```
Input Video (MP4)
      |
      v
[VideoReader] -- reads frames sequentially
      |
      v
[Detector] -- YOLOv11m, class=0 ("person"), confidence >= 0.25
      |  Output: sv.Detections (bbox_xyxy, confidence, class_id)
      v
[Tracker] -- BoT-SORT: Kalman + IoU + ReID + Camera Motion Compensation
      |  Output: sv.Detections with persistent tracker_id
      v
[Annotator] -- bounding boxes + ID labels + trajectory trails
      |
      v
[VideoWriter] -- annotated output MP4
```

## Technical Approach

### Detection: YOLOv11m
- Pretrained on COCO dataset, filtered to class 0 (person)
- Medium variant balances accuracy and speed
- Hardware-accelerated inference on Apple Silicon (MPS) or NVIDIA GPU (CUDA)

### Tracking: BoT-SORT
Selected over simpler alternatives (ByteTrack, DeepSORT) because sports footage demands:

1. **ReID Appearance Embeddings** -- When players in identical uniforms cross paths, appearance features (512-dim vectors from OSNet) maintain correct ID associations
2. **Camera Motion Compensation (GMC)** -- Sports broadcasts have constant panning/zooming; BoT-SORT estimates global motion via sparse optical flow and adjusts Kalman filter predictions
3. **Two-Stage Association** -- High-confidence detections matched first by IoU + appearance, then low-confidence detections matched to remaining tracks (recovers partially occluded players)
4. **Track Lifecycle Management** -- New tracks confirmed after 3 consecutive frames (prevents false positive IDs); lost tracks kept for 30 frames for ReID re-association

### Why This Combination?
- YOLOv11m provides reliable person detection at high FPS
- BoT-SORT addresses every challenge listed in the assignment: occlusion, motion blur, scale changes, camera motion, similar appearances
- The `supervision` library provides clean annotation without custom OpenCV drawing code

## Installation

### Prerequisites
- Python 3.9+
- ffmpeg (for video encoding)
- yt-dlp (for video download)

### Setup
```bash
# Clone the repository
git clone https://github.com/Manishsingh85/image-processing-pipeline.git
cd imageProcessingPipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv11 detection models |
| `boxmot` | BoT-SORT tracker with ReID |
| `supervision` | Annotation and visualization |
| `opencv-python` | Video I/O and image processing |
| `torch` | Deep learning backend |
| `matplotlib` / `seaborn` | Charts and heatmaps |
| `yt-dlp` | Public video download |

## Usage

### 1. Download a Public Video
```bash
python scripts/download_video.py --url "YOUR_YOUTUBE_URL" --duration 60
```

### 2. Run the Pipeline
```bash
# Default: YOLOv11m + BoT-SORT
python scripts/run_pipeline.py --input data/input/cricket.mp4

# With options
python scripts/run_pipeline.py \
    --input data/input/cricket.mp4 \
    --model yolo11l.pt \
    --tracker botsort \
    --confidence 0.3 \
    --device mps
```

### 3. Generate Report Assets
```bash
python scripts/generate_report_assets.py --input data/input/cricket.mp4
```

### 4. Jupyter Notebook Walkthrough
```bash
jupyter notebook notebooks/pipeline_demo.ipynb
```

## Project Structure

```
imageProcessingPipeline/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore
├── config/
│   └── settings.py                 # All configurable parameters
├── src/
│   ├── detector.py                 # YOLOv11 detection wrapper
│   ├── tracker.py                  # BoT-SORT / ByteTrack tracking
│   ├── annotator.py                # Frame annotation (boxes, labels, trails)
│   ├── pipeline.py                 # Main pipeline orchestrator
│   ├── video_io.py                 # Video reader/writer utilities
│   └── utils.py                    # Color palette, helpers
├── enhancements/
│   ├── trajectory.py               # Trajectory trail visualization
│   ├── heatmap.py                  # Movement heatmap generation
│   ├── object_count.py             # Object count over time chart
│   └── model_comparison.py         # Multi-model/tracker comparison
├── notebooks/
│   ├── pipeline_demo.ipynb         # Step-by-step pipeline walkthrough
│   └── analysis.ipynb              # Post-run analysis and charts
├── scripts/
│   ├── run_pipeline.py             # CLI entry point
│   ├── download_video.py           # Video download + trim
│   └── generate_report_assets.py   # Report figure generation
├── report/
│   ├── technical_report.md         # Detailed technical report
│   └── figures/                    # Charts, heatmaps, screenshots
└── data/
    ├── input/                      # Original input videos
    ├── output/                     # Annotated output videos
    └── screenshots/                # Pipeline screenshots
```

## Optional Enhancements Implemented

| Enhancement | Description |
|---|---|
| **Trajectory Visualization** | Color-coded movement trails behind each tracked subject |
| **Movement Heatmap** | Spatial density map showing where players spend the most time |
| **Object Count Over Time** | Time-series chart of tracked subjects per frame |
| **Model Comparison** | Side-by-side comparison of YOLOv11 n/m/l with BoT-SORT vs ByteTrack |

## Assumptions & Limitations

### Assumptions
- Input video is broadcast-quality sports footage (720p+)
- Primary subjects to track are people (COCO class 0)
- Camera is relatively stable (broadcast camera, not handheld)

### Limitations
- ReID model is pretrained on general pedestrian datasets, not sports-specific -- fine-tuning on sports data would improve ID consistency
- Very fast camera cuts (scene changes) will cause ID loss -- the tracker handles smooth pans/zooms but not hard cuts
- Players at very small scale (distant fielders in cricket) may not be detected consistently
- BoT-SORT is slower than ByteTrack due to ReID extraction -- ~5-15 FPS vs ~20-40 FPS depending on hardware
- Track IDs are not guaranteed to be sequential -- gaps occur when tentative tracks are discarded

### Failure Cases
- Complete occlusion for >3 seconds -- track is terminated, new ID assigned on re-emergence
- Players swapping positions while fully overlapping -- ReID may confuse identities if appearance is identical
- Rapid zoom transitions -- GMC may not fully compensate, causing brief ID switches

## Video Source

**Original video link:** [INSERT PUBLIC VIDEO URL HERE]

The video was downloaded and trimmed using `scripts/download_video.py`. Source metadata is stored in `data/input/video_metadata.json`.

## References

- [YOLOv11 (Ultralytics)](https://docs.ultralytics.com/)
- [BoT-SORT Paper](https://arxiv.org/abs/2206.14651)
- [boxmot Library](https://github.com/mikel-brostrom/boxmot)
- [supervision Library](https://github.com/roboflow/supervision)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
