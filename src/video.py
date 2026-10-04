# pyrefly: ignore [missing-import]
import cv2
import time
from typing import Tuple, Optional, Generator
from src.config import get_config
from src.logger import logger
import numpy as np


class VideoStream:
    def __init__(self):
        self.config = get_config()
        self.source_type = self.config.input.source_type

        if self.source_type == "video":
            self.source = self.config.input.video_path
        elif self.source_type == "rtsp":
            self.source = self.config.input.rtsp_url
        else:
            raise ValueError(f"Unknown source_type: {self.source_type}")

        # Clamp frame_skip to at least 1 to prevent ZeroDivisionError
        self.frame_skip = max(1, self.config.detection.frame_skip)
        self.cap = None
        self.frame_count = 0

    def start(self):
        """Open the video stream."""
        logger.info(f"Opening video source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise RuntimeError(f"Could not open video source: {self.source}")

        # For RTSP, reduce buffer size to minimize latency
        if self.source_type == "rtsp":
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        logger.info("Video source opened successfully.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        self.frame_count += 1
        return ret, frame

    def generate_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields (frame_id, frame) tuples.
        Respects the frame_skip config: only yields every N-th frame
        (frame_count % frame_skip == 0) to reduce CPU load, while
        still advancing the capture so we stay in real-time sync.
        """
        self.start()

        try:
            while True:
                ret, frame = self.read()

                if not ret:
                    logger.info("End of video stream or connection lost.")
                    break

                # Skip frames to match desired processing rate
                if self.frame_count % self.frame_skip != 0:
                    continue

                yield self.frame_count, frame

        finally:
            self.stop()

    def stop(self):
        """Release video capture resources (idempotent — safe to call multiple times)."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Video stream released.")
        self.cap = None
