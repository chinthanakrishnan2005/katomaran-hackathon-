import sqlite3
import os
import json
from datetime import datetime
from typing import List, Tuple, Optional
from src.config import get_config
from src.logger import logger

class Database:
    def __init__(self):
        config = get_config()
        self.db_path = config.database.path
        
        # Ensure directory exists (guard against empty dirname)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self._init_db()

    def _get_connection(self):
        # We create a new connection per operation to avoid thread issues, 
        # but for a simple sequential pipeline a persistent one is also fine.
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # VISITORS Table
                # We store embedding as a JSON string for simplicity, or we could use BLOB.
                # Since sqlite doesn't easily search vectors, we just store it for record keeping,
                # and maintain the actual gallery in memory.
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS VISITORS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT UNIQUE NOT NULL,
                        embedding_path TEXT,
                        first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # EVENTS Table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS EVENTS (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        face_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        image_path TEXT,
                        FOREIGN KEY (face_id) REFERENCES VISITORS(face_id)
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_id ON EVENTS(face_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON EVENTS(event_type)')
                
                conn.commit()
                logger.debug("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    def register_visitor(self, face_id: str, embedding_path: str = "") -> bool:
        """Register a new visitor."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO VISITORS (face_id, embedding_path) VALUES (?, ?)',
                    (face_id, embedding_path)
                )
                conn.commit()
                logger.info(f"Registered new visitor: {face_id}")
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Visitor {face_id} already exists in database.")
            return False
        except Exception as e:
            logger.error(f"Failed to register visitor {face_id}: {e}")
            return False

    def log_event(self, face_id: str, event_type: str, image_path: str = "") -> bool:
        """Log an entry or exit event."""
        if event_type not in ['ENTRY', 'EXIT']:
            logger.error(f"Invalid event type: {event_type}")
            return False
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO EVENTS (face_id, event_type, image_path) VALUES (?, ?, ?)',
                    (face_id, event_type, image_path)
                )
                conn.commit()
                logger.info(f"Logged {event_type} event for {face_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to log event for {face_id}: {e}")
            return False

    def get_unique_visitor_count(self) -> int:
        """Return the total number of unique visitors registered."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM VISITORS')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to get visitor count: {e}")
            return 0
            
    def get_all_visitors(self) -> List[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT face_id FROM VISITORS')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get visitors: {e}")
            return []
