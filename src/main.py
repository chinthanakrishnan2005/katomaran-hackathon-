import warnings
# Suppress FutureWarnings from third-party libraries (insightface, scikit-image)
# These do not affect functionality and would otherwise cause PowerShell to report exit code 1
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import time
from src.config import get_config
from src.logger import logger
from src.database import Database
from src.video import VideoStream
from src.tracker import FaceTracker
from src.recognizer import FaceRecognizer
from src.event_logger import EventLogger


def main():
    config = get_config()
    logger.info("Starting Intelligent Face Tracker...")

    # --- Initialize components ---
    db = Database()
    event_logger = EventLogger(db)
    video = VideoStream()
    tracker = FaceTracker(model_path="yolov8n.pt", conf_thresh=0.5)
    recognizer = FaceRecognizer()

    # Grace periods
    grace_period = 30   # frames of absence before EXIT is triggered
    entry_grace = 5     # frames a track must be stable before ENTRY is logged

    # Set of track_ids that have already had an ENTRY event logged
    processed_entries: set = set()

    start_time = time.time()
    total_frames = 0

    try:
        for frame_id, frame in video.generate_frames():
            total_frames += 1

            # --- Detection & Tracking ---
            active_tracks = tracker.track(frame, frame_id)

            # --- Recognition & ENTRY logic ---
            for track in active_tracks:
                # Cache the latest frame on the track for EXIT event use
                track.last_frame = frame.copy()

                # Only try recognition after the track has been stable for `entry_grace` frames
                if track.face_id is None and (frame_id - track.first_seen_frame) >= entry_grace:
                    embedding = recognizer.extract_embedding(frame, track.face_info)

                    if embedding is not None:
                        face_id, is_new = recognizer.recognize(embedding)
                        track.face_id = face_id

                        if is_new:
                            db.register_visitor(face_id)

                # Log ENTRY once we have a confirmed face_id
                if track.face_id and track.track_id not in processed_entries:
                    event_logger.process_entry(frame, track)
                    processed_entries.add(track.track_id)

            # --- EXIT logic ---
            for tid, state in list(tracker.active_tracks.items()):
                if state.frames_since_last_seen > grace_period:
                    if state.face_id and tid in processed_entries:
                        # Use the cached last-seen frame so the crop is meaningful
                        exit_frame = state.last_frame if state.last_frame is not None else frame
                        event_logger.process_exit(exit_frame, state)
                    # Remove the expired track from active tracking
                    del tracker.active_tracks[tid]

            # --- Visualization ---
            if config.output.visualize:
                display_frame = frame.copy()
                for track in active_tracks:
                    bbox = track.face_info.bbox
                    x1 = int(bbox.x1)
                    y1 = int(bbox.y1)
                    x2 = int(bbox.x2)
                    y2 = int(bbox.y2)

                    label = f"TID:{track.track_id}"
                    if track.face_id:
                        label += f" | {track.face_id}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame, label,
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

                # Display unique visitor count overlay
                unique_count = db.get_unique_visitor_count()
                cv2.putText(
                    display_frame, f"Unique Visitors: {unique_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

                cv2.imshow("Face Tracker", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User pressed 'q' — stopping.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        # video.stop() is also called by the generator's finally block, but our
        # stop() is idempotent so this is safe.
        video.stop()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0
        logger.info(
            f"Shutdown complete. Processed {total_frames} frames in "
            f"{elapsed:.2f}s ({fps:.1f} FPS)."
        )
        logger.info(f"Total Unique Visitors: {db.get_unique_visitor_count()}")


if __name__ == "__main__":
    main()
