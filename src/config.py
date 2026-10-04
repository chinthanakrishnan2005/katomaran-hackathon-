import json
import os
from dataclasses import dataclass

@dataclass
class InputConfig:
    source_type: str
    video_path: str
    rtsp_url: str

@dataclass
class DetectionConfig:
    frame_skip: int

@dataclass
class RecognitionConfig:
    similarity_threshold: float

@dataclass
class DatabaseConfig:
    path: str

@dataclass
class LoggingConfig:
    event_log: str

@dataclass
class OutputConfig:
    visualize: bool
    entry_images_dir: str
    exit_images_dir: str

class AppConfig:
    def __init__(self, config_path: str = "config.json"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data = json.load(f)
            
        self.input = InputConfig(**data.get("input", {}))
        self.detection = DetectionConfig(**data.get("detection", {}))
        self.recognition = RecognitionConfig(**data.get("recognition", {}))
        self.database = DatabaseConfig(**data.get("database", {}))
        self.logging = LoggingConfig(**data.get("logging", {}))
        self.output = OutputConfig(**data.get("output", {}))

# Singleton instance to be imported by other modules
# Will be initialized when first imported. We can wrap it in a function if lazy load is needed.
# For simplicity, we can load it on import or provide a load_config function.
_config_instance = None

def get_config(config_path: str = "config.json") -> AppConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig(config_path)
    return _config_instance
