import sys
import os
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.video import VideoStream
from src.detector import FaceDetector

def test_phase3_4():
    video_stream = VideoStream()
    detector = FaceDetector(model_path="yolov8n.pt")
    
    frame_count = 0
    for frame_id, frame in video_stream.generate_frames():
        if frame_count > 10:  # just test first 10 frames
            break
            
        if frame_id % 5 == 0:
            faces = detector.detect(frame)
            print(f"Frame {frame_id}: detected {len(faces)} person(s)")
            
        frame_count += 1
        
    print("Test Phase 3 & 4 passed!")

if __name__ == "__main__":
    test_phase3_4()
