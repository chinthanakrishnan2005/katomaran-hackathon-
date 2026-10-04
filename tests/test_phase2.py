import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_config
from src.logger import logger
from src.database import Database

def test_phase2():
    config = get_config()
    print("Config loaded:")
    print("Video path:", config.input.video_path)
    print("Database path:", config.database.path)
    
    logger.info("Testing logger...")
    
    db = Database()
    db.register_visitor("Face_001")
    db.register_visitor("Face_001") # Should warn, not error
    db.log_event("Face_001", "ENTRY")
    
    count = db.get_unique_visitor_count()
    print("Unique visitors:", count)

if __name__ == "__main__":
    test_phase2()
