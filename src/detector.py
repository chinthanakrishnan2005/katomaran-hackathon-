from ultralytics import YOLO
import numpy as np
import os
from typing import List
from src.models import FaceInfo, BBox
from src.logger import logger

class FaceDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_thresh: float = 0.5):
        # We try to use a face model if provided, otherwise fallback to yolov8n.pt
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        
        logger.info(f"Loading YOLO model from {model_path}...")
        try:
            # If the user has pip installed yolov8face, maybe they have a different way.
            # But normally ultralytics YOLO class can load it if it's a valid weights file.
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[FaceInfo]:
        """Detect faces in a frame."""
        # YOLOv8 inference
        results = self.model(frame, verbose=False, conf=self.conf_thresh)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # If we are using standard yolov8n.pt, class 0 is person.
                # If we are using a face model, there usually is only class 0 (face).
                cls_id = int(box.cls[0].item())
                
                # Assuming class 0 is what we want (person/face)
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].item())
                    
                    bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=conf)
                    faces.append(FaceInfo(bbox=bbox))
                    
        return faces
