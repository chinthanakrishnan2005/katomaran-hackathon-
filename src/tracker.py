from ultralytics import YOLO
import numpy as np
from typing import List, Dict
from src.models import FaceInfo, BBox, TrackState
from src.logger import logger

class FaceTracker:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        logger.info(f"Initializing ByteTrack Tracker with YOLO model {model_path}...")
        self.model = YOLO(model_path)
        
        # Track active states
        self.active_tracks: Dict[int, TrackState] = {}
        
    def track(self, frame: np.ndarray, frame_id: int) -> List[TrackState]:
        """
        Run detection and tracking on the frame.
        """
        # Run tracking using ByteTrack
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=self.conf_thresh)
        
        current_frame_tracks = []
        
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                # Ensure there is a track ID assigned
                if box.id is None:
                    continue
                    
                track_id = int(box.id[0].item())
                cls_id = int(box.cls[0].item())
                
                # Class 0 is person/face
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    
                    bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=conf)
                    face_info = FaceInfo(bbox=bbox)
                    
                    if track_id not in self.active_tracks:
                        # New track
                        self.active_tracks[track_id] = TrackState(
                            track_id=track_id, 
                            face_info=face_info,
                            first_seen_frame=frame_id,
                            last_seen_frame=frame_id
                        )
                    else:
                        # Update existing track
                        state = self.active_tracks[track_id]
                        state.face_info.bbox = bbox  # update bbox
                        state.last_seen_frame = frame_id
                        state.frames_since_last_seen = 0
                        state.is_active = True
                        
                    current_frame_tracks.append(self.active_tracks[track_id])
                    
        # Update missing tracks
        for tid, state in list(self.active_tracks.items()):
            if state.last_seen_frame < frame_id:
                state.frames_since_last_seen += 1
                
        return current_frame_tracks
