"""
Deep learning models for sign language gesture recognition
Implements CNN, CNN-LSTM, and Transformer architectures
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import logging

from config.config import config

class TensorFlowModels:
    """TensorFlow/Keras implementations of gesture recognition models"""
    
    @staticmethod
    def create_cnn_model(input_shape: Tuple[int, int, int], num_classes: int) -> Model:
        """
        Create a CNN model for single frame gesture recognition
        
        Args:
            input_shape: Shape of input images (H, W, C)
            num_classes: Number of gesture classes
            
        Returns:
            Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling and dense layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='gesture_cnn')
        return model
    
    @staticmethod
    def create_cnn_lstm_model(
        input_shape: Tuple[int, int, int, int], 
        num_classes: int
    ) -> Model:
        """
        Create a CNN-LSTM model for sequence-based gesture recognition
        
        Args:
            input_shape: Shape of input sequences (T, H, W, C)
            num_classes: Number of gesture classes
            
        Returns:
            Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # CNN feature extractor (applied to each frame)
        cnn_base = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu')
        ])
        
        # Apply CNN to each frame in the sequence
        x = layers.TimeDistributed(cnn_base)(inputs)
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, dropout=0.3)(x)
        
        # Final classification layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='gesture_cnn_lstm')
        return model
    
    @staticmethod
    def create_transformer_model(
        input_shape: Tuple[int, int], 
        num_classes: int,
        num_heads: int = 8,
        d_model: int = 256
    ) -> Model:
        """
        Create a Transformer model for landmark-based gesture recognition
        
        Args:
            input_shape: Shape of input landmarks (T, features)
            num_classes: Number of gesture classes
            num_heads: Number of attention heads
            d_model: Model dimension
            
        Returns:
            Keras model
        """
        inputs = keras.Input(shape=input_shape)
        
        # Position encoding
        x = layers.Dense(d_model)(inputs)
        
        # Multi-head attention blocks
        for _ in range(4):  # 4 transformer blocks
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=d_model // num_heads
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization()(x)
            
            # Feed-forward network
            ff_output = layers.Dense(d_model * 2, activation='relu')(x)
            ff_output = layers.Dense(d_model)(ff_output)
            
            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization()(x)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='gesture_transformer')
        return model

class PyTorchModels:
    """PyTorch implementations of gesture recognition models"""
    
    class CNNLSTMModel(nn.Module):
        """PyTorch CNN-LSTM model for gesture recognition"""
        
        def __init__(self, num_classes: int, input_channels: int = 3):
            super().__init__()
            
            # CNN feature extractor
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU()
            )
            
            # LSTM for temporal modeling
            self.lstm = nn.LSTM(256, 128, batch_first=True, dropout=0.3)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            # x shape: (batch, sequence, channels, height, width)
            batch_size, seq_len = x.size(0), x.size(1)
            
            # Process each frame through CNN
            cnn_features = []
            for i in range(seq_len):
                frame_features = self.cnn(x[:, i])
                cnn_features.append(frame_features)
            
            # Stack features for LSTM
            cnn_features = torch.stack(cnn_features, dim=1)
            
            # LSTM processing
            lstm_out, _ = self.lstm(cnn_features)
            
            # Use last LSTM output for classification
            final_features = lstm_out[:, -1]
            
            # Classification
            output = self.classifier(final_features)
            
            return output

class ModelFactory:
    """Factory class for creating gesture recognition models"""
    
    @staticmethod
    def create_model(
        model_type: str,
        input_shape: Tuple,
        num_classes: int,
        framework: str = "tensorflow"
    ):
        """
        Create a gesture recognition model
        
        Args:
            model_type: Type of model ('cnn', 'cnn_lstm', 'transformer')
            input_shape: Shape of input data
            num_classes: Number of gesture classes
            framework: Deep learning framework ('tensorflow' or 'pytorch')
            
        Returns:
            Model instance
        """
        if framework == "tensorflow":
            if model_type == "cnn":
                return TensorFlowModels.create_cnn_model(input_shape, num_classes)
            elif model_type == "cnn_lstm":
                return TensorFlowModels.create_cnn_lstm_model(input_shape, num_classes)
            elif model_type == "transformer":
                return TensorFlowModels.create_transformer_model(input_shape, num_classes)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        elif framework == "pytorch":
            if model_type == "cnn_lstm":
                return PyTorchModels.CNNLSTMModel(num_classes)
            else:
                raise ValueError(f"PyTorch model type {model_type} not implemented")
        
        else:
            raise ValueError(f"Unknown framework: {framework}")

class ModelTrainer:
    """Training utilities for gesture recognition models"""
    
    def __init__(self, model, model_type: str = "tensorflow"):
        self.model = model
        self.model_type = model_type
        self.history = None
    
    def compile_model(self, learning_rate: float = None):
        """Compile the model with appropriate settings"""
        if learning_rate is None:
            learning_rate = config.model.learning_rate
        
        if self.model_type == "tensorflow":
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
    
    def train(
        self,
        train_data,
        validation_data,
        epochs: int = None,
        callbacks: List = None
    ):
        """Train the model"""
        if epochs is None:
            epochs = config.model.epochs
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        if self.model_type == "tensorflow":
            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return self.history
    
    def _get_default_callbacks(self):
        """Get default training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=config.model.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=config.model.checkpoint_path + 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        return callbacks

# Utility functions
def get_model_summary(model, framework: str = "tensorflow"):
    """Get model summary information"""
    if framework == "tensorflow":
        return model.summary()
    elif framework == "pytorch":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"Total parameters: {total_params}, Trainable: {trainable_params}"

def save_model(model, filepath: str, framework: str = "tensorflow"):
    """Save trained model"""
    if framework == "tensorflow":
        model.save(filepath)
    elif framework == "pytorch":
        torch.save(model.state_dict(), filepath)
    
    logging.info(f"Model saved to {filepath}")

def load_model(filepath: str, model_class=None, framework: str = "tensorflow"):
    """Load trained model"""
    if framework == "tensorflow":
        return keras.models.load_model(filepath)
    elif framework == "pytorch":
        model = model_class
        model.load_state_dict(torch.load(filepath))
        return model