# Technical Report: Multi-Object Detection & Persistent ID Tracking

## 1. Introduction

This report describes a computer vision pipeline for detecting and tracking multiple subjects in public sports video footage. The pipeline assigns persistent unique IDs to each detected person, maintaining identity consistency through real-world challenges including occlusion, camera motion, scale variation, and visually similar subjects (e.g., players in identical team uniforms).

## 2. Methodology

### 2.1 Object Detection: YOLOv11m

The detection stage uses **YOLOv11m** (medium) from the Ultralytics framework, pretrained on the COCO dataset. We filter detections to class 0 ("person") with a confidence threshold of 0.25.

**Why YOLOv11m:**
- **Speed-accuracy tradeoff:** The medium variant provides strong detection accuracy (~50 mAP on COCO) while maintaining real-time inference speeds on GPU/MPS hardware.
- **Ultralytics ecosystem:** Seamless integration with the `supervision` library for annotation and with boxmot for tracking.
- **Low-confidence recovery:** Setting the threshold to 0.25 (rather than the typical 0.5) allows partially occluded players to be detected, which the tracker can then associate via its two-stage matching.

### 2.2 Multi-Object Tracking: BoT-SORT

The tracking stage uses **BoT-SORT** (Bottom-Up Object Tracking with Spatial-temporal Re-identification) via the boxmot library. BoT-SORT was selected as the primary tracker based on the specific challenges outlined in the assignment.

**BoT-SORT combines four key components:**

1. **Kalman Filter Motion Model:** Predicts the next-frame position of each track based on constant-velocity assumption. This provides the primary motion prior for data association.

2. **Two-Stage IoU Association (Hungarian Algorithm):**
   - First pass: high-confidence detections (>0.5) matched to predicted track positions via IoU.
   - Second pass: remaining low-confidence detections (>0.1) matched to unmatched tracks. This ByteTrack-inspired approach recovers partially occluded subjects that would be missed by single-threshold methods.

3. **ReID Appearance Embeddings:** A lightweight OSNet ReID model (`osnet_x0_25_msmt17`) extracts 512-dimensional appearance feature vectors for each detection. When IoU-based matching is ambiguous (e.g., two players crossing paths), cosine similarity between appearance embeddings disambiguates the correct identity. When a track is "lost" for several frames and re-appears, appearance matching re-links the original ID rather than assigning a new one.

4. **Camera Motion Compensation (GMC):** Sports broadcasts involve constant camera panning, tilting, and zooming. Without compensation, these global motions cause the Kalman filter's predicted positions to diverge from actual positions, leading to mass ID switches. BoT-SORT estimates the inter-frame global motion using sparse optical flow and applies a homography correction to Kalman filter predictions before association.

**Why BoT-SORT over ByteTrack:**
ByteTrack is faster (no ReID overhead) but relies solely on IoU and motion for association. In sports footage where:
- Players wear identical uniforms (IoU alone cannot distinguish them)
- Camera pans frequently (pure Kalman predictions drift)
- Players exit and re-enter the frame (no appearance memory)

...BoT-SORT's ReID + GMC combination provides measurably better ID consistency. Our model comparison confirms this: BoT-SORT produces fewer unique IDs (indicating fewer ID switches) than ByteTrack on the same video.

### 2.3 Pipeline Architecture

The pipeline follows a modular design:

```
VideoReader → Detector → Tracker → Annotator → VideoWriter
                                       ↓
                              EnhancementAccumulator
                                       ↓
                          Heatmap / Count Chart / Comparison
```

All modules communicate via `supervision.Detections` objects, providing a clean, standardized interface. The pipeline processes every frame sequentially (no frame skipping) to ensure tracking continuity.

## 3. ID Consistency Maintenance

ID consistency is the central technical challenge. Our approach uses a 5-layer strategy:

| Layer | Component | Handles |
|---|---|---|
| 1 | Kalman Filter | Smooth linear motion |
| 2 | IoU Association | Frame-to-frame spatial overlap |
| 3 | ReID Embeddings | Similar appearance disambiguation |
| 4 | Camera Motion Compensation | Global camera panning/zooming |
| 5 | Track Lifecycle | Tentative/lost/dead state management |

**Track lifecycle parameters:**
- New tracks require 3 consecutive frames of confirmation before receiving a visible ID (prevents false positive IDs from noise).
- Lost tracks are kept in memory for 30 frames (~1 second at 30 FPS) for potential ReID re-association.
- Tracks with no match for 90 frames (~3 seconds) are permanently terminated.

## 4. Challenges Faced

1. **Similar Uniforms:** Cricket/football players on the same team are visually near-identical. The pretrained ReID model (trained on pedestrian datasets) struggles with sports-specific appearance variation. Fine-tuning on sports data would improve this.

2. **Small-Scale Detections:** Distant fielders in wide-angle cricket shots are very small (< 30px height). YOLOv11m occasionally misses these, causing track gaps.

3. **Camera Cuts:** Hard scene cuts (e.g., replay transitions) cause all tracks to be lost. The pipeline handles smooth camera motion well but cannot bridge discontinuous cuts.

4. **MPS Compatibility:** Some ReID models have limited support for Apple Silicon's MPS backend. We run ReID on CPU as a reliable fallback while YOLO inference uses MPS.

## 5. Failure Cases

- **Complete occlusion >3 seconds:** If a player is fully hidden behind another for more than the track buffer (30 frames), the track is terminated. Upon re-emergence, a new ID is assigned.
- **Simultaneous crossing of identical players:** When two players with nearly identical appearance cross paths while partially occluded, ReID similarity scores may be too close to disambiguate, resulting in an ID swap.
- **Broadcast overlays:** Score tickers, replay graphics, and watermarks can occasionally be detected as objects or interfere with tracking near frame edges.

## 6. Model Comparison

We compared three YOLOv11 variants (nano, medium, large) with two trackers (BoT-SORT, ByteTrack) on the same video (first 300 frames):

| Metric | Key Finding |
|---|---|
| **FPS** | YOLOv11n+ByteTrack is fastest; YOLOv11l+BoT-SORT is slowest |
| **Unique IDs** | BoT-SORT consistently produces fewer unique IDs (better ID consistency) than ByteTrack |
| **Detections/Frame** | Larger models detect more persons per frame, especially distant/small subjects |

The medium model with BoT-SORT offers the best balance of speed, detection coverage, and ID stability.

## 7. Possible Improvements

1. **Sports-Specific ReID Fine-Tuning:** Train the ReID model on sports datasets (e.g., SoccerNet, SportsMOT) to better distinguish players by jersey number, body type, and equipment rather than general pedestrian features.

2. **Transformer-Based Tracking:** Models like MOTRv2 or TrackFormer perform joint detection and tracking in an end-to-end manner, potentially reducing ID switches in complex scenarios.

3. **Scene Cut Detection:** Implement hard-cut detection (e.g., histogram difference between frames) to reset the tracker cleanly during replays/transitions rather than generating spurious tracks.

4. **Team Clustering:** Use jersey color extraction (dominant color in bounding box) to automatically cluster players into teams, adding semantic meaning to track IDs.

5. **Bird's-Eye View Projection:** Apply homography transformation to project player positions onto a top-down field view, enabling tactical analysis and more accurate speed estimation.

6. **Temporal Smoothing:** Apply smoothing to bounding box trajectories to reduce jitter from frame-to-frame detection noise, producing visually cleaner output.
