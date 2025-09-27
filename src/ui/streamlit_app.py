"""
Streamlit-based web interface for real-time sign language recognition
Provides user-friendly interface for gesture-to-text translation
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import threading
from collections import deque
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional

# Import our custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import MediaPipeDetector
from src.models.nlp_processor import GestureToTextMapper
from config.config import config

class RealTimeGestureRecognizer:
    """Real-time gesture recognition system"""
    
    def __init__(self):
        self.detector = MediaPipeDetector()
        self.nlp_processor = GestureToTextMapper()
        self.is_running = False
        self.gesture_buffer = deque(maxlen=config.ui.prediction_buffer_size)
        self.current_sentence = ""
        self.confidence_history = deque(maxlen=50)
        
        # Mock model for demonstration (replace with actual trained model)
        self.mock_predictions = {
            'hello': 0.95,
            'thank_you': 0.88,
            'please': 0.82,
            'yes': 0.91,
            'no': 0.87,
            'help': 0.79,
            'water': 0.84,
            'food': 0.86
        }
    
    def predict_gesture(self, landmarks: Dict) -> tuple:
        """
        Predict gesture from landmarks (mock implementation)
        Replace this with actual model inference
        """
        # Mock prediction based on hand presence
        if landmarks['hands']:
            # Simulate gesture prediction
            import random
            gesture = random.choice(list(self.mock_predictions.keys()))
            confidence = self.mock_predictions[gesture] + random.uniform(-0.1, 0.1)
            confidence = max(0.0, min(1.0, confidence))
            return gesture, confidence
        
        return None, 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame and return predictions"""
        # Extract landmarks
        landmarks = self.detector.extract_landmarks(frame)
        
        # Predict gesture
        gesture, confidence = self.predict_gesture(landmarks)
        
        # Store confidence history
        self.confidence_history.append(confidence)
        
        result = {
            'gesture': gesture,
            'confidence': confidence,
            'landmarks': landmarks,
            'timestamp': time.time()
        }
        
        # Add to buffer if confident enough
        if confidence >= config.ui.prediction_threshold:
            self.gesture_buffer.append((gesture, confidence))
        
        return result

def main():
    st.set_page_config(
        page_title="Sign Language to Text Translator",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .gesture-display {
        font-size: 1.5rem;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high {
        background-color: #d4edda;
        color: #155724;
    }
    .confidence-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .confidence-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">👋 Gestures to Phrases: Sign Language Translator</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = RealTimeGestureRecognizer()
    
    if 'translated_text' not in st.session_state:
        st.session_state.translated_text = ""
    
    if 'gesture_history' not in st.session_state:
        st.session_state.gesture_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Language selection
        target_language = st.selectbox(
            "Select Target Language",
            options=config.nlp.supported_languages,
            format_func=lambda x: {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 
                'de': 'German', 'it': 'Italian'
            }.get(x, x)
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=1.0,
            value=config.ui.prediction_threshold,
            step=0.05,
            help="Minimum confidence for gesture recognition"
        )
        
        # Update config
        config.ui.prediction_threshold = confidence_threshold
        
        # Clear buttons
        if st.button("🗑️ Clear Text"):
            st.session_state.translated_text = ""
            st.session_state.recognizer.nlp_processor.clear_context()
        
        if st.button("📊 Clear History"):
            st.session_state.gesture_history = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Camera Feed")
        
        # Camera input
        camera_input = st.camera_input("Take a picture for gesture recognition")
        
        if camera_input is not None:
            # Convert to OpenCV format
            image = Image.open(camera_input)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Process frame
            result = st.session_state.recognizer.process_frame(frame)
            
            # Display results
            if result['gesture']:
                # Get text mapping
                text_result = st.session_state.recognizer.nlp_processor.map_gesture_to_text(
                    result['gesture'], 
                    result['confidence']
                )
                
                # Update translated text
                if result['confidence'] >= confidence_threshold:
                    if st.session_state.translated_text:
                        st.session_state.translated_text += " " + text_result['phrase']
                    else:
                        st.session_state.translated_text = text_result['phrase']
                    
                    # Add to history
                    st.session_state.gesture_history.append({
                        'timestamp': time.strftime("%H:%M:%S"),
                        'gesture': result['gesture'],
                        'confidence': result['confidence'],
                        'text': text_result['phrase']
                    })
                
                # Display current gesture
                confidence_class = (
                    "confidence-high" if result['confidence'] >= 0.8 
                    else "confidence-medium" if result['confidence'] >= 0.6 
                    else "confidence-low"
                )
                
                st.markdown(f"""
                <div class="gesture-display {confidence_class}">
                    <strong>Detected:</strong> {result['gesture']}<br>
                    <strong>Confidence:</strong> {result['confidence']:.2%}<br>
                    <strong>Text:</strong> {text_result['phrase']}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("💬 Translation Output")
        
        # Display translated text
        text_area = st.text_area(
            "Translated Text",
            value=st.session_state.translated_text,
            height=200,
            placeholder="Your translated text will appear here..."
        )
        
        # Update if manually edited
        if text_area != st.session_state.translated_text:
            st.session_state.translated_text = text_area
        
        # Translation to other languages
        if target_language != 'en' and st.session_state.translated_text:
            translated = st.session_state.recognizer.nlp_processor.translate_text(
                st.session_state.translated_text, 
                target_language
            )
            st.text_area(
                f"Translation ({target_language.upper()})",
                value=translated,
                height=100
            )
        
        # Context information
        context_summary = st.session_state.recognizer.nlp_processor.get_context_summary()
        if not context_summary.get('empty', True):
            st.subheader("🧠 Context")
            st.write(f"**Recent words:** {', '.join(context_summary['recent_words'])}")
            st.write(f"**Category:** {context_summary['dominant_category']}")
            st.write(f"**Avg. Confidence:** {context_summary['average_confidence']:.2%}")
    
    # Performance metrics and history
    st.subheader("📊 Performance & History")
    
    tab1, tab2, tab3 = st.tabs(["📈 Real-time Metrics", "📋 Gesture History", "ℹ️ System Info"])
    
    with tab1:
        if st.session_state.recognizer.confidence_history:
            # Confidence chart
            confidence_data = list(st.session_state.recognizer.confidence_history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=confidence_data,
                mode='lines+markers',
                name='Confidence',
                line=dict(color='blue', width=2)
            ))
            fig.add_hline(y=confidence_threshold, line_dash="dash", 
                         line_color="red", annotation_text="Threshold")
            fig.update_layout(
                title="Real-time Confidence Score",
                yaxis_title="Confidence",
                xaxis_title="Time",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No confidence data available yet. Start recognizing gestures!")
    
    with tab2:
        if st.session_state.gesture_history:
            # Display recent gestures
            df = pd.DataFrame(st.session_state.gesture_history[-20:])  # Last 20 gestures
            st.dataframe(
                df[['timestamp', 'gesture', 'confidence', 'text']],
                use_container_width=True
            )
            
            # Gesture frequency chart
            gesture_counts = df['gesture'].value_counts()
            fig = px.bar(
                x=gesture_counts.index,
                y=gesture_counts.values,
                title="Most Frequent Gestures",
                labels={'x': 'Gesture', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No gesture history available yet.")
    
    with tab3:
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Supported Gestures", len(config.data.asl_classes))
            st.metric("Languages Supported", len(config.nlp.supported_languages))
            st.metric("Current Threshold", f"{confidence_threshold:.2%}")
        
        with col2:
            st.metric("Model Type", config.model.model_type.upper())
            st.metric("Input Resolution", f"{config.data.frame_width}x{config.data.frame_height}")
            st.metric("Buffer Size", config.ui.prediction_buffer_size)
        
        # Supported gestures
        st.subheader("📚 Supported Gestures")
        gesture_categories = {
            'Greetings': ['hello', 'good'],
            'Politeness': ['thank_you', 'please', 'sorry'],
            'Responses': ['yes', 'no'],
            'Needs': ['help', 'water', 'food', 'more'],
            'Emotions': ['happy', 'sad'],
            'Places': ['home', 'work', 'school'],
            'Alphabet': list('ABCDEFGHIJ')  # First 10 letters as example
        }
        
        for category, gestures in gesture_categories.items():
            with st.expander(f"{category} ({len(gestures)} gestures)"):
                st.write(", ".join(gestures))
    
    # Footer with tips
    st.markdown("---")
    st.subheader("💡 Tips for Better Recognition")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        **Camera & Lighting:**
        - Ensure good lighting on your hands
        - Keep hands within camera frame
        - Avoid busy backgrounds
        - Maintain steady hand positions
        """)
    
    with tips_col2:
        st.markdown("""
        **Gesture Performance:**
        - Make clear, deliberate gestures
        - Hold each gesture for 1-2 seconds
        - Practice standard ASL hand shapes
        - Speak gestures in sequence for sentences
        """)

if __name__ == "__main__":
    main()