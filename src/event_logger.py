import os
import cv2
import numpy as np
from datetime import datetime
from src.database import Database
from src.config import get_config
from src.logger import logger
from src.models import TrackState


class EventLogger:
    def __init__(self, db: Database):
        self.db = db
        self.config = get_config()
        self.entry_dir = self.config.output.entry_images_dir
        self.exit_dir = self.config.output.exit_images_dir

    def _save_image(self, frame: np.ndarray, track: TrackState, event_type: str) -> str:
        """
        Crop and save a face image from `frame` using the track's bounding box.
        Returns the saved filepath, or empty string on failure.
        """
        if frame is None or frame.size == 0:
            logger.warning(f"Empty frame passed to _save_image for {event_type} event.")
            return ""

        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S-%f")[:12]

        base_dir = self.entry_dir if event_type == "ENTRY" else self.exit_dir
        dir_path = os.path.join(base_dir, date_str)
        os.makedirs(dir_path, exist_ok=True)

        face_id = track.face_id if track.face_id else f"Unknown_Track_{track.track_id}"
        filename = f"{face_id}_{event_type}_{time_str}.jpg"
        filepath = os.path.join(dir_path, filename)

        # Crop the face using the track's bounding box (with a small margin)
        bbox = track.face_info.bbox
        h, w = frame.shape[:2]

        margin_x = int((bbox.x2 - bbox.x1) * 0.1)
        margin_y = int((bbox.y2 - bbox.y1) * 0.1)

        x1 = max(0, int(bbox.x1) - margin_x)
        y1 = max(0, int(bbox.y1) - margin_y)
        x2 = min(w, int(bbox.x2) + margin_x)
        y2 = min(h, int(bbox.y2) + margin_y)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            cv2.imwrite(filepath, crop)
            logger.debug(f"Saved {event_type} image: {filepath}")
            return filepath

        logger.warning(f"Crop for {event_type} event is empty (bbox out of frame?).")
        return ""

    def process_entry(self, frame: np.ndarray, track: TrackState):
        """Process and log a new ENTRY event."""
        if not track.face_id:
            logger.warning("Attempted to log ENTRY for a track without a face_id.")
            return

        img_path = self._save_image(frame, track, "ENTRY")
        track.last_image_path = img_path

        self.db.log_event(track.face_id, "ENTRY", img_path)
        logger.info(f"ENTRY logged for {track.face_id} (track {track.track_id})")

    def process_exit(self, frame: np.ndarray, track: TrackState):
        """Process and log an EXIT event."""
        if not track.face_id:
            return

        img_path = self._save_image(frame, track, "EXIT")
        self.db.log_event(track.face_id, "EXIT", img_path)
        logger.info(f"EXIT logged for {track.face_id} (track {track.track_id})")
