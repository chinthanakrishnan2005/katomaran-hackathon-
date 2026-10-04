from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class FaceInfo:
    bbox: BBox
    landmark: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class TrackState:
    track_id: int
    face_info: FaceInfo
    face_id: Optional[str] = None
    is_active: bool = True
    frames_since_last_seen: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_image_path: Optional[str] = None
    # Cache the last good frame so EXIT events can save a meaningful crop
    last_frame: Optional[np.ndarray] = field(default=None, repr=False)
