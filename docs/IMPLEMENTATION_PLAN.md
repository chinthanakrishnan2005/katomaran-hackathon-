# Implementation Plan

## Requirements
- Real-time video/RTSP processing for face tracking.
- YOLO-based face detection (e.g., YOLOv8 face).
- InsightFace-based face recognition and embedding extraction.
- Automatic registration of new faces.
- ByteTrack or similar approach for stable multi-object tracking.
- Entry/Exit logging to SQLite database.
- Unique visitor counting.
- Configurable settings via `config.json`.
- Event logging to `logs/events.log`.
- Saving cropped images of entries and exits.

## Architecture
- **Input Source**: Configurable (Video file or RTSP stream).
- **Video Capture Module**: Reads frames and applies a configurable frame-skip.
- **Detector Module (YOLO)**: Detects faces and yields bounding boxes with confidence.
- **Tracker Module (ByteTrack)**: Maintains track IDs across frames to handle temporary occlusion.
- **Recognizer Module (InsightFace)**: Extracts 512-d embeddings from cropped faces and matches against registered identities in memory (using cosine similarity/L2 distance). Handles automatic registration if no match is found.
- **Event Logger Module**: Handles generation of entry and exit events. Defines entry as the start of a track and exit as the end of a track (after a timeout).
- **Database Module (SQLite)**: Stores `VISITORS` and `EVENTS` records. 
- **Main Pipeline**: Ties the components together, orchestrating the data flow and UI overlay (if enabled).

## Modules
1. `config.py`: Loads and validates `config.json`.
2. `database.py`: SQLite connection and schema setup.
3. `logger.py`: Configures standard Python `logging`.
4. `models.py`: Defines data classes (e.g., `Face`, `Track`, `Event`).
5. `detector.py`: YOLOv8 face detection wrapper.
6. `tracker.py`: ByteTrack integration.
7. `recognizer.py`: InsightFace integration and identity management.
8. `event_logger.py`: Event lifecycle (entry, exit, image saving).
9. `main.py`: Entry point and processing loop.

## Data Flow
Video / RTSP -> Frame Capture -> Frame Skipping -> YOLO Face Detection -> Object Tracking (ByteTrack) -> For each track: Face Crop / Alignment -> InsightFace Embedding -> Face Registration / Existing Face Match -> Entry / Exit Detection -> Image + Event Logger -> SQLite Database -> Unique Visitor Count.

## Database Design
- **VISITORS Table**:
  - `id` (INTEGER PRIMARY KEY)
  - `face_id` (TEXT UNIQUE)
  - `embedding` (BLOB or stored locally in a numpy file and referenced)
  - `first_seen` (DATETIME)
  - `created_at` (DATETIME)

- **EVENTS Table**:
  - `id` (INTEGER PRIMARY KEY)
  - `face_id` (TEXT)
  - `event_type` (TEXT - 'ENTRY' or 'EXIT')
  - `timestamp` (DATETIME)
  - `image_path` (TEXT)

## Tracking Strategy
- Use ByteTrack to track detected face bounding boxes.
- Assign an internal `track_id` to each continuous bounding box.
- The `track_id` is linked to a permanent `face_id` via the Recognizer.

## Face Recognition Strategy
- Maintain an active gallery of embeddings in memory (e.g., NumPy array or Faiss).
- Compare new embeddings using cosine similarity. If similarity > threshold (e.g., 0.45-0.5), it's a match.
- If it's a new identity, register it as `Face_XXX`, save the embedding, and add to the gallery.

## Entry/Exit Strategy
- **Entry**: When a new `track_id` is confidently identified and stabilized for a few frames, generate an ENTRY event.
- **Exit**: When a `track_id` is lost for more than a configured `grace_period` (e.g., 30 frames), generate an EXIT event.
- Each `face_id` generates at most one entry and one exit per continuous track. If they return later, they generate a new entry.

## Unique Visitor Counting Strategy
- Count the number of distinct `face_id`s in the `VISITORS` table.
- Maintain a running count in memory.

## Testing Strategy
- Unit tests for configuration logic, database operations, recognition distance calculations, and event generation logic.
- E2E testing using a provided sample video.

## Dependency Plan
- `opencv-python`: For video and image processing.
- `ultralytics`: For YOLOv8.
- `insightface`: For ArcFace embedding generation.
- `onnxruntime`: Backend for InsightFace.
- `lapx` & `scipy`: For tracker matching (ByteTrack).
- `sqlite3`: Built-in python DB.

## Implementation Phases
1. Audit & Architecture (Current Phase)
2. Config, Database, Logging
3. Video/RTSP Input
4. YOLO Detection
5. InsightFace Recognition & Registration
6. Tracking
7. Entry/Exit Logic
8. Unique Visitor Counting
9. Integration
10. E2E Testing
11. Fixes
12. Documentation
13. Final Audit
