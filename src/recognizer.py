import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from typing import Optional, Tuple, Dict
from src.models import FaceInfo
from src.config import get_config
from src.logger import logger
import cv2

class FaceRecognizer:
    def __init__(self):
        config = get_config()
        self.threshold = config.recognition.similarity_threshold
        
        logger.info("Initializing InsightFace (buffalo_l)...")
        # Initialize FaceAnalysis. It will auto-download models to ~/.insightface if missing.
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace initialized.")
        
        # In-memory gallery: face_id -> embedding (np.ndarray)
        self.gallery: Dict[str, np.ndarray] = {}
        self.next_id = 1

    def load_gallery(self, database_records):
        """
        Load embeddings from the database. 
        For now, we simulate this as the DB only stores paths or we can assume it starts empty.
        """
        # If we stored embeddings as numpy files, we could load them here.
        # But for this implementation, we will start with an empty gallery 
        # and auto-register as we go. In a production app, we'd load saved .npy files.
        pass

    def extract_embedding(self, frame: np.ndarray, face_info: FaceInfo) -> Optional[np.ndarray]:
        """
        Given a full frame and a YOLO-detected face_info (bounding box), 
        crop the face and use InsightFace to extract the embedding.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face_info.bbox.x1, face_info.bbox.y1, face_info.bbox.x2, face_info.bbox.y2
        
        # Add some margin to the bounding box for InsightFace to work better
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        
        cx1 = max(0, int(x1) - margin_x)
        cy1 = max(0, int(y1) - margin_y)
        cx2 = min(w, int(x2) + margin_x)
        cy2 = min(h, int(y2) + margin_y)
        
        crop = frame[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            return None
            
        # Run InsightFace on the crop
        faces = self.app.get(crop)
        
        if not faces:
            return None
            
        # Take the largest face in the crop
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        face = faces[0]
        
        # Save embedding to our face_info
        face_info.embedding = face.embedding
        return face.embedding

    def recognize(self, embedding: np.ndarray) -> Tuple[str, bool]:
        """
        Match embedding against gallery.
        Returns: (face_id, is_new_registration)
        """
        best_match_id = None
        best_similarity = -1.0
        
        # Compare with all registered faces
        for face_id, reg_emb in self.gallery.items():
            # Cosine similarity = 1 - cosine distance
            sim = 1.0 - cosine(embedding, reg_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_match_id = face_id
                
        if best_similarity >= self.threshold and best_match_id is not None:
            return best_match_id, False
            
        # If no match found or below threshold, register a new identity
        new_id = f"Face_{self.next_id:03d}"
        self.next_id += 1
        
        self.gallery[new_id] = embedding
        logger.info(f"Auto-registered new face: {new_id} (sim: {best_similarity:.2f})")
        return new_id, True
