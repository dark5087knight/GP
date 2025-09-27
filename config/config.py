"""
Configuration settings for the Sign Language Recognition System
"""
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    # Model architecture
    model_type: str = "cnn_lstm"  # Options: 'cnn', 'cnn_lstm', 'transformer'
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    sequence_length: int = 30  # Number of frames for video sequences
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Model paths
    model_save_path: str = "data/models/"
    checkpoint_path: str = "data/models/checkpoints/"

@dataclass
class DataConfig:
    """Data processing configuration"""
    # Data paths
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    
    # Video processing
    fps: int = 30
    frame_width: int = 224
    frame_height: int = 224
    
    # Data augmentation
    rotation_range: float = 10.0
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    
    # Sign language classes (expandable)
    asl_classes: List[str] = None
    
    def __post_init__(self):
        if self.asl_classes is None:
            # Common ASL signs to start with
            self.asl_classes = [
                'hello', 'thank_you', 'please', 'yes', 'no', 'good', 'bad',
                'help', 'water', 'food', 'more', 'finish', 'sorry', 'love',
                'family', 'friend', 'home', 'work', 'school', 'happy',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'  # Letters
            ]

@dataclass
class UIConfig:
    """User interface configuration"""
    # Camera settings
    camera_index: int = 0
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.5
    
    # Display settings
    window_width: int = 1280
    window_height: int = 720
    fps_display: bool = True
    
    # Real-time processing
    prediction_threshold: float = 0.8
    smooth_predictions: bool = True
    prediction_buffer_size: int = 5

@dataclass
class NLPConfig:
    """Natural Language Processing configuration"""
    # Language models
    default_language: str = "en"
    supported_languages: List[str] = None
    
    # Text processing
    max_sentence_length: int = 100
    context_window: int = 5  # Number of previous gestures for context
    
    # Translation services
    use_offline_translation: bool = True
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "it"]

# Global configuration instance
class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.ui = UIConfig()
        self.nlp = NLPConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.model.model_save_path,
            self.model.checkpoint_path,
            self.data.raw_data_path,
            self.data.processed_data_path
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()