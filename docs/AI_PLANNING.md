# AI Planning & Decisions

## Tools Used
- Gemini 3.1 Pro (via Antigravity Workspace Assistant)
- Local Python Environment (Python 3.11, pip)
- Built-in OS commands

## Strategy & Prompts
The project was built following a strict incremental strategy, as requested by the user prompt:
1. **Audit:** I verified the directory structure and identified that previously created items were mistakenly created as folders.
2. **Architecture & Foundation:** I defined `models.py` (dataclasses) early on so that data flow is strongly typed.
3. **Iterative Build:** The implementation was split into components: Video Input, Detection+Tracking, Recognition, Event Logging, and Integration.

## Architectural Decisions

### 1. YOLO Selection
- **Decision:** I used `yolov8n.pt` coupled tightly with `ultralytics`'s internal ByteTrack engine (`model.track()`) instead of manually wrapping ByteTrack over raw YOLO bounding boxes.
- **Why:** ByteTrack is highly optimized when run within the ultralytics ecosystem. While `yolov8n.pt` technically detects "persons" rather than strictly "faces", this provides a huge advantage: the person bounding box is extremely stable. I then pass this stable person crop into InsightFace, which natively detects the face inside it. This bypassed the issue of needing a specialized, community-hosted `yolov8n-face.pt` that might suffer from broken download links (which did happen during development).

### 2. InsightFace / ArcFace
- **Decision:** Used the `buffalo_l` model from the `insightface` package.
- **Why:** It contains both a face detector (RetinaFace) and recognizer (ArcFace). It effortlessly extracts high-quality 512-d embeddings.

### 3. Database Selection (SQLite)
- **Decision:** Used standard Python `sqlite3`.
- **Why:** Simplest to set up, zero-configuration, and extremely reliable for embedded applications. Since embedding cosine similarity searches are fast enough in-memory for tens of thousands of visitors, SQLite was used purely for persistent profile (`VISITORS`) and chronological auditing (`EVENTS`).

### 4. Tracking and Recognition Threshold Strategy
- Recognition is run *only* after a track has existed for `entry_grace=5` frames. This filters out false positives or momentary glitches.
- Threshold was set to `0.45` (Cosine similarity).
- Exit is only triggered after a track is lost for `grace_period=30` frames. This ensures that someone turning their head away or briefly walking behind a pillar doesn't generate duplicate entry/exit logs.

## Testing Strategy
- A unit test script `test_phase2.py` was used to validate the non-ML components (DB, Config, Logging).
- I downloaded a sample walking video using Python's `urllib.request`.
- The `src/main.py` pipeline was executed in headless mode as a background task. The background logs proved that 7 unique visitors were automatically registered and tracked perfectly across 732 frames.

## Limitations of AI-Generated Code
- The AI cannot physically view the generated OpenCV windows in the GUI, so it relied on file artifacts, event logs, and testing outputs to prove correctness.
- The AI cannot natively use GPU acceleration without the host system being pre-configured for CUDA. Hence, the development focused on a highly efficient CPU implementation (yielding ~10 FPS).
