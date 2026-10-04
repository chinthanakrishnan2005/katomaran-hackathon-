# Intelligent Face Tracker

## Problem Statement
The objective is to build an Intelligent Face Tracker with Auto-Registration and Visitor Counting. The system must process a video stream, detect faces, automatically register new faces, recognize them later, track them across frames, log entry/exit events, store timestamped cropped face images, store metadata in a database, and maintain an accurate unique visitor count.

## Features
- Real-time video (MP4) and RTSP stream processing.
- YOLOv8 + ByteTrack for highly stable multi-object tracking.
- InsightFace (ArcFace) for SOTA face embedding and recognition.
- Zero-shot automatic registration of new identities.
- Robust Entry/Exit event logging with occlusion handling (grace periods).
- Local SQLite database to persist visitor history and unique visitor counts.
- Configurable settings via `config.json`.

## Technology Stack
- **Python 3.11+**
- **Ultralytics (YOLOv8)**
- **InsightFace**
- **OpenCV**
- **SQLite**

## Project Structure
```text
katomaran-face-tracker/
├── src/
│   ├── main.py            # Orchestrator
│   ├── config.py          # JSON config loader
│   ├── database.py        # SQLite interactions
│   ├── detector.py        # (Legacy) YOLO Wrapper
│   ├── tracker.py         # YOLO + ByteTrack tracker
│   ├── recognizer.py      # InsightFace embeddings & matching
│   ├── event_logger.py    # Image cropping and DB event routing
│   ├── models.py          # Dataclasses
│   └── logger.py          # System logging configuration
├── config.json            # Application configuration
├── requirements.txt       # Dependencies
├── data/                  # Contains sample.mp4
├── logs/                  # Contains events.log and entry/exit images
├── database/              # SQLite files
└── docs/                  # Architecture & Planning docs
```

## Installation & Environment Setup

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: InsightFace and YOLO will automatically download their required model weights `.pt` / `.onnx` upon first execution).*

## Configuration

Modify `config.json` in the root directory:
```json
{
    "input": {
        "source_type": "video",
        "video_path": "data/sample.mp4",
        "rtsp_url": "rtsp://username:password@ip_address/stream"
    },
    "detection": {
        "frame_skip": 5
    },
    "recognition": {
        "similarity_threshold": 0.45
    },
    "database": {
        "path": "database/visitors.db"
    },
    "output": {
        "visualize": true
    }
}
```

## How to Run

1. **Video Input Usage:**
   Set `"source_type": "video"` in `config.json` and ensure `"video_path"` points to a valid file.
   ```bash
   python -m src.main
   ```

2. **RTSP Usage:**
   Change `"source_type"` to `"rtsp"` and provide your camera stream URL in `"rtsp_url"`.
   ```bash
   python -m src.main
   ```

## Database Structure
Stored locally in `database/visitors.db` using SQLite.
- `VISITORS`: Contains `id`, `face_id` (e.g., Face_001), `embedding_path`, `first_seen`.
- `EVENTS`: Contains `id`, `face_id`, `event_type` (ENTRY/EXIT), `timestamp`, `image_path`.

## System Logic
### Entry/Exit Logic
- **Entry:** Triggered when a new track ID has remained active and stable for `5` frames.
- **Exit:** Triggered when a tracked individual goes out of frame or is fully occluded for `30` frames. This ensures brief disappearances don't create spam logs.
- Images are captured dynamically from the bounding box upon these events and saved to `logs/entries/YYYY-MM-DD/`.

### Unique Visitor Counting
Unique visitor count is calculated by running a `SELECT COUNT(*)` on the `VISITORS` table. Since duplicates are caught using Cosine Similarity thresholds in `recognizer.py`, the count remains strictly accurate.

## Testing & Performance
The system was successfully executed end-to-end on a local CPU processing `732 frames` in `~70 seconds` yielding `10+ FPS`.
See [COMPUTE_ESTIMATION.md](docs/COMPUTE_ESTIMATION.md) for more details.

## Troubleshooting
- **No faces detected:** Check your `config.json` similarity threshold. Lower it slightly if the environment lighting is poor.
- **RTSP Lag:** Ensure you are on a fast network. Set `frame_skip` to `2` or `3` to allow the CPU to catch up on dense crowds.

---
This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com/)
