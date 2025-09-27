"""
Desktop GUI application for sign language recognition using tkinter
Provides native desktop interface with real-time camera feed
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from collections import deque
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocessor import MediaPipeDetector
from src.models.nlp_processor import GestureToTextMapper
from config.config import config

class SignLanguageGUI:
    """Main GUI application for sign language recognition"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text Translator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.detector = MediaPipeDetector()
        self.nlp_processor = GestureToTextMapper()
        
        # Camera and recognition variables
        self.cap = None
        self.is_camera_active = False
        self.current_frame = None
        self.gesture_buffer = deque(maxlen=config.ui.prediction_buffer_size)
        self.recognition_thread = None
        
        # UI Variables
        self.translated_text = tk.StringVar()
        self.current_gesture = tk.StringVar(value="No gesture detected")
        self.confidence_var = tk.StringVar(value="0.00")
        self.fps_var = tk.StringVar(value="0")
        
        # Create UI
        self.create_widgets()
        self.setup_camera()
        
        # Mock model for demonstration
        self.mock_predictions = {
            'hello': 0.95, 'thank_you': 0.88, 'please': 0.82,
            'yes': 0.91, 'no': 0.87, 'help': 0.79,
            'water': 0.84, 'food': 0.86, 'more': 0.83,
            'sorry': 0.89, 'family': 0.76, 'home': 0.81
        }
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="👋 Sign Language to Text Translator",
            font=('Arial', 18, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Camera and controls
        left_frame = ttk.LabelFrame(main_frame, text="Camera & Recognition", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera feed
        self.camera_label = ttk.Label(left_frame, text="Camera initializing...")
        self.camera_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Camera controls
        camera_controls = ttk.Frame(left_frame)
        camera_controls.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(
            camera_controls, 
            text="Start Camera", 
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.capture_button = ttk.Button(
            camera_controls, 
            text="Capture Gesture", 
            command=self.capture_gesture,
            state=tk.DISABLED
        )
        self.capture_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Recognition info
        info_frame = ttk.LabelFrame(left_frame, text="Recognition Info", padding="5")
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))\n        \n        # Current gesture\n        ttk.Label(info_frame, text=\"Current Gesture:\").grid(row=0, column=0, sticky=tk.W)\n        gesture_label = ttk.Label(\n            info_frame, \n            textvariable=self.current_gesture,\n            font=('Arial', 12, 'bold'),\n            foreground='blue'\n        )\n        gesture_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))\n        \n        # Confidence\n        ttk.Label(info_frame, text=\"Confidence:\").grid(row=1, column=0, sticky=tk.W)\n        confidence_label = ttk.Label(info_frame, textvariable=self.confidence_var)\n        confidence_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))\n        \n        # FPS\n        ttk.Label(info_frame, text=\"FPS:\").grid(row=2, column=0, sticky=tk.W)\n        fps_label = ttk.Label(info_frame, textvariable=self.fps_var)\n        fps_label.grid(row=2, column=1, sticky=tk.W, padx=(10, 0))\n        \n        # Right panel - Text output and controls\n        right_frame = ttk.LabelFrame(main_frame, text=\"Translation Output\", padding=\"10\")\n        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))\n        right_frame.columnconfigure(0, weight=1)\n        right_frame.rowconfigure(1, weight=1)\n        \n        # Language selection\n        lang_frame = ttk.Frame(right_frame)\n        lang_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))\n        \n        ttk.Label(lang_frame, text=\"Target Language:\").pack(side=tk.LEFT)\n        self.language_var = tk.StringVar(value=\"en\")\n        language_combo = ttk.Combobox(\n            lang_frame,\n            textvariable=self.language_var,\n            values=[(\"en\", \"English\"), (\"es\", \"Spanish\"), (\"fr\", \"French\")],\n            state=\"readonly\",\n            width=15\n        )\n        language_combo.pack(side=tk.LEFT, padx=(10, 0))\n        \n        # Text output area\n        self.text_output = scrolledtext.ScrolledText(\n            right_frame,\n            height=15,\n            width=50,\n            font=('Arial', 12),\n            wrap=tk.WORD\n        )\n        self.text_output.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))\n        \n        # Text controls\n        text_controls = ttk.Frame(right_frame)\n        text_controls.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))\n        \n        ttk.Button(\n            text_controls, \n            text=\"Clear Text\", \n            command=self.clear_text\n        ).pack(side=tk.LEFT)\n        \n        ttk.Button(\n            text_controls, \n            text=\"Copy Text\", \n            command=self.copy_text\n        ).pack(side=tk.LEFT, padx=(10, 0))\n        \n        ttk.Button(\n            text_controls, \n            text=\"Save Text\", \n            command=self.save_text\n        ).pack(side=tk.LEFT, padx=(10, 0))\n        \n        # Settings panel\n        settings_frame = ttk.LabelFrame(right_frame, text=\"Settings\", padding=\"5\")\n        settings_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))\n        \n        # Confidence threshold\n        ttk.Label(settings_frame, text=\"Confidence Threshold:\").grid(row=0, column=0, sticky=tk.W)\n        self.threshold_var = tk.DoubleVar(value=config.ui.prediction_threshold)\n        threshold_scale = ttk.Scale(\n            settings_frame,\n            from_=0.5,\n            to=1.0,\n            variable=self.threshold_var,\n            orient=tk.HORIZONTAL,\n            length=200\n        )\n        threshold_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))\n        \n        threshold_label = ttk.Label(settings_frame, textvariable=self.threshold_var)\n        threshold_label.grid(row=0, column=2, padx=(10, 0))\n    \n    def setup_camera(self):\n        \"\"\"Initialize camera\"\"\"\n        try:\n            self.cap = cv2.VideoCapture(config.ui.camera_index)\n            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.ui.window_width // 2)\n            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ui.window_height // 2)\n        except Exception as e:\n            messagebox.showerror(\"Camera Error\", f\"Failed to initialize camera: {e}\")\n    \n    def toggle_camera(self):\n        \"\"\"Toggle camera on/off\"\"\"\n        if self.is_camera_active:\n            self.stop_camera()\n        else:\n            self.start_camera()\n    \n    def start_camera(self):\n        \"\"\"Start camera and recognition\"\"\"\n        if self.cap is None:\n            self.setup_camera()\n        \n        if self.cap and self.cap.isOpened():\n            self.is_camera_active = True\n            self.start_button.config(text=\"Stop Camera\")\n            self.capture_button.config(state=tk.NORMAL)\n            \n            # Start camera thread\n            self.recognition_thread = threading.Thread(target=self.camera_loop, daemon=True)\n            self.recognition_thread.start()\n        else:\n            messagebox.showerror(\"Camera Error\", \"Could not start camera\")\n    \n    def stop_camera(self):\n        \"\"\"Stop camera and recognition\"\"\"\n        self.is_camera_active = False\n        self.start_button.config(text=\"Start Camera\")\n        self.capture_button.config(state=tk.DISABLED)\n        \n        if self.cap:\n            self.cap.release()\n            self.cap = None\n        \n        # Clear camera display\n        self.camera_label.config(image='', text=\"Camera stopped\")\n    \n    def camera_loop(self):\n        \"\"\"Main camera processing loop\"\"\"\n        fps_counter = 0\n        fps_start_time = time.time()\n        \n        while self.is_camera_active:\n            if self.cap and self.cap.isOpened():\n                ret, frame = self.cap.read()\n                if ret:\n                    # Flip frame horizontally for mirror effect\n                    frame = cv2.flip(frame, 1)\n                    self.current_frame = frame.copy()\n                    \n                    # Process frame for gesture recognition\n                    self.process_frame(frame)\n                    \n                    # Convert frame for display\n                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n                    display_frame = cv2.resize(display_frame, (480, 360))\n                    \n                    # Convert to PIL and then to PhotoImage\n                    pil_image = Image.fromarray(display_frame)\n                    photo = ImageTk.PhotoImage(pil_image)\n                    \n                    # Update display in main thread\n                    self.root.after(0, self.update_camera_display, photo)\n                    \n                    # Calculate FPS\n                    fps_counter += 1\n                    if time.time() - fps_start_time >= 1.0:\n                        fps = fps_counter / (time.time() - fps_start_time)\n                        self.root.after(0, lambda: self.fps_var.set(f\"{fps:.1f}\"))\n                        fps_counter = 0\n                        fps_start_time = time.time()\n                    \n                    # Small delay to prevent overwhelming the GUI\n                    time.sleep(0.03)  # ~30 FPS\n            else:\n                break\n    \n    def update_camera_display(self, photo):\n        \"\"\"Update camera display in main thread\"\"\"\n        self.camera_label.config(image=photo, text=\"\")\n        self.camera_label.image = photo  # Keep a reference\n    \n    def process_frame(self, frame):\n        \"\"\"Process frame for gesture recognition\"\"\"\n        try:\n            # Extract landmarks\n            landmarks = self.detector.extract_landmarks(frame)\n            \n            # Mock gesture prediction\n            gesture, confidence = self.predict_gesture(landmarks)\n            \n            # Update UI\n            self.root.after(0, self.update_recognition_info, gesture, confidence)\n            \n            # Add to buffer if confident enough\n            if confidence >= self.threshold_var.get():\n                self.gesture_buffer.append((gesture, confidence))\n                \n                # Generate text if we have enough gestures\n                if len(self.gesture_buffer) >= 3:\n                    self.root.after(0, self.update_text_output)\n        \n        except Exception as e:\n            print(f\"Error processing frame: {e}\")\n    \n    def predict_gesture(self, landmarks):\n        \"\"\"Mock gesture prediction\"\"\"\n        if landmarks['hands']:\n            import random\n            gesture = random.choice(list(self.mock_predictions.keys()))\n            base_confidence = self.mock_predictions[gesture]\n            confidence = base_confidence + random.uniform(-0.1, 0.1)\n            confidence = max(0.0, min(1.0, confidence))\n            return gesture, confidence\n        return \"none\", 0.0\n    \n    def update_recognition_info(self, gesture, confidence):\n        \"\"\"Update recognition information display\"\"\"\n        if gesture and gesture != \"none\":\n            self.current_gesture.set(gesture.replace('_', ' ').title())\n            self.confidence_var.set(f\"{confidence:.2%}\")\n        else:\n            self.current_gesture.set(\"No gesture detected\")\n            self.confidence_var.set(\"0.00%\")\n    \n    def update_text_output(self):\n        \"\"\"Update text output based on gesture buffer\"\"\"\n        if len(self.gesture_buffer) >= 3:\n            # Get recent gestures\n            recent_gestures = list(self.gesture_buffer)[-3:]\n            \n            # Generate sentence\n            sentence = self.nlp_processor.generate_sentence(recent_gestures)\n            \n            if sentence:\n                # Add to text output\n                self.text_output.insert(tk.END, sentence + \" \")\n                self.text_output.see(tk.END)\n                \n                # Clear buffer\n                self.gesture_buffer.clear()\n    \n    def capture_gesture(self):\n        \"\"\"Manually capture and process current gesture\"\"\"\n        if self.current_frame is not None:\n            # Process current frame\n            landmarks = self.detector.extract_landmarks(self.current_frame)\n            gesture, confidence = self.predict_gesture(landmarks)\n            \n            if gesture and gesture != \"none\" and confidence >= self.threshold_var.get():\n                # Get text mapping\n                text_result = self.nlp_processor.map_gesture_to_text(gesture, confidence)\n                \n                # Add to text output\n                self.text_output.insert(tk.END, text_result['phrase'] + \" \")\n                self.text_output.see(tk.END)\n            else:\n                messagebox.showinfo(\"Capture\", \"No confident gesture detected\")\n    \n    def clear_text(self):\n        \"\"\"Clear text output\"\"\"\n        self.text_output.delete(1.0, tk.END)\n        self.nlp_processor.clear_context()\n    \n    def copy_text(self):\n        \"\"\"Copy text to clipboard\"\"\"\n        text = self.text_output.get(1.0, tk.END).strip()\n        if text:\n            self.root.clipboard_clear()\n            self.root.clipboard_append(text)\n            messagebox.showinfo(\"Copied\", \"Text copied to clipboard\")\n    \n    def save_text(self):\n        \"\"\"Save text to file\"\"\"\n        from tkinter import filedialog\n        \n        text = self.text_output.get(1.0, tk.END).strip()\n        if text:\n            filename = filedialog.asksaveasfilename(\n                defaultextension=\".txt\",\n                filetypes=[(\"Text files\", \"*.txt\"), (\"All files\", \"*.*\")]\n            )\n            if filename:\n                try:\n                    with open(filename, 'w', encoding='utf-8') as f:\n                        f.write(text)\n                    messagebox.showinfo(\"Saved\", f\"Text saved to {filename}\")\n                except Exception as e:\n                    messagebox.showerror(\"Error\", f\"Failed to save file: {e}\")\n    \n    def on_closing(self):\n        \"\"\"Handle application closing\"\"\"\n        self.stop_camera()\n        if self.detector:\n            self.detector.close()\n        self.root.destroy()\n\ndef main():\n    \"\"\"Main application entry point\"\"\"\n    root = tk.Tk()\n    app = SignLanguageGUI(root)\n    \n    # Handle window closing\n    root.protocol(\"WM_DELETE_WINDOW\", app.on_closing)\n    \n    # Start the GUI\n    root.mainloop()\n\nif __name__ == \"__main__\":\n    main()