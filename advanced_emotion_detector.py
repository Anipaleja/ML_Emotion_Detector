#!/usr/bin/env python
"""
Advanced Emotion Detector - Enhanced version of the emotion recognition system

This module provides advanced capabilities for emotion recognition:
1. Real-time webcam processing
2. Multi-face tracking and analysis
3. Emotion history tracking and trend analysis
4. Advanced visualization
5. Confidence scoring

It can be used as a standalone application or integrated with the API service.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque, defaultdict
import threading
import queue
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.emotion_model import EmotionRecognitionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
HISTORY_LENGTH = 100  # Number of frames to keep in history
CONFIDENCE_THRESHOLD = 0.60  # Minimum confidence to consider a prediction reliable
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "saved_model", "emotion_model")
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red (BGR)
    'disgust': (0, 128, 0),    # Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 255, 0),   # Light Green
    'neutral': (128, 128, 128) # Gray
}

class FaceTracker:
    """Track individual faces across frames with unique IDs"""
    
    def __init__(self, max_distance=0.6, max_disappeared=30):
        self.next_face_id = 0
        self.faces = {}  # Dictionary of tracked faces
        self.disappeared = {}  # Number of frames a face has disappeared
        self.max_distance = max_distance  # Maximum distance for face matching
        self.max_disappeared = max_disappeared  # Maximum number of frames a face can disappear before being removed
        
    def register(self, face_encoding, face_location):
        """Register a new face"""
        face_id = self.next_face_id
        self.faces[face_id] = {
            "encoding": face_encoding,
            "location": face_location,
            "last_seen": time.time()
        }
        self.disappeared[face_id] = 0
        self.next_face_id += 1
        return face_id
        
    def deregister(self, face_id):
        """Remove a face from tracking"""
        del self.faces[face_id]
        del self.disappeared[face_id]
        
    def update(self, face_encodings, face_locations):
        """Update tracked faces with new detections"""
        # If no faces in current frame, mark all as disappeared
        if len(face_encodings) == 0:
            for face_id in list(self.disappeared.keys()):
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
            return {}
            
        # If we're not tracking any faces yet, register all
        if len(self.faces) == 0:
            face_ids = {}
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                face_id = self.register(encoding, location)
                face_ids[i] = face_id
            return face_ids
            
        # Try to match existing faces with new detections
        face_ids = {}
        used_faces = set()
        
        for i, (new_encoding, new_location) in enumerate(zip(face_encodings, face_locations)):
            min_distance = float('inf')
            matched_face_id = None
            
            for face_id, face_data in self.faces.items():
                if face_id in used_faces:
                    continue
                    
                # Calculate face similarity
                distance = np.linalg.norm(np.array(face_data["encoding"]) - np.array(new_encoding))
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    matched_face_id = face_id
            
            if matched_face_id is not None:
                # Update existing face
                self.faces[matched_face_id]["encoding"] = new_encoding
                self.faces[matched_face_id]["location"] = new_location
                self.faces[matched_face_id]["last_seen"] = time.time()
                self.disappeared[matched_face_id] = 0
                face_ids[i] = matched_face_id
                used_faces.add(matched_face_id)
            else:
                # Register new face
                face_id = self.register(new_encoding, new_location)
                face_ids[i] = face_id
                
        # Mark faces that weren't matched as disappeared
        for face_id in list(self.disappeared.keys()):
            if face_id not in used_faces:
                self.disappeared[face_id] += 1
                if self.disappeared[face_id] > self.max_disappeared:
                    self.deregister(face_id)
                    
        return face_ids


class EmotionHistory:
    """Track emotion history for each face"""
    
    def __init__(self, history_length=HISTORY_LENGTH):
        self.history = defaultdict(lambda: {
            "timestamps": deque(maxlen=history_length),
            "emotions": deque(maxlen=history_length),
            "confidences": deque(maxlen=history_length)
        })
        
    def add_emotion(self, face_id, timestamp, emotion, confidence):
        """Add emotion prediction to history"""
        self.history[face_id]["timestamps"].append(timestamp)
        self.history[face_id]["emotions"].append(emotion)
        self.history[face_id]["confidences"].append(confidence)
        
    def get_dominant_emotion(self, face_id, window=10):
        """Get the dominant emotion over the last window frames"""
        if face_id not in self.history or len(self.history[face_id]["emotions"]) == 0:
            return None, 0.0
            
        # Get most recent emotions up to window size
        recent_emotions = list(self.history[face_id]["emotions"])[-window:]
        if not recent_emotions:
            return None, 0.0
            
        # Count occurrences
        emotion_counts = {}
        for emotion in recent_emotions:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
                
        # Find most common
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        confidence = emotion_counts[dominant_emotion] / len(recent_emotions)
        
        return dominant_emotion, confidence
        
    def get_emotion_trend(self, face_id, window=30):
        """Analyze emotion trends for a face"""
        if face_id not in self.history or len(self.history[face_id]["emotions"]) < window:
            return {}
            
        # Get emotions in the window
        recent_emotions = list(self.history[face_id]["emotions"])[-window:]
        emotion_counts = {}
        
        for emotion in recent_emotions:
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
                
        # Convert to percentages
        total = len(recent_emotions)
        emotion_percentages = {emotion: count/total for emotion, count in emotion_counts.items()}
        
        # Detect shifts in emotion
        first_half = recent_emotions[:window//2]
        second_half = recent_emotions[window//2:]
        
        first_half_counts = {}
        for emotion in first_half:
            if emotion in first_half_counts:
                first_half_counts[emotion] += 1
            else:
                first_half_counts[emotion] = 1
                
        second_half_counts = {}
        for emotion in second_half:
            if emotion in second_half_counts:
                second_half_counts[emotion] += 1
            else:
                second_half_counts[emotion] = 1
                
        # Calculate trend direction for each emotion
        trends = {}
        for emotion in set(list(first_half_counts.keys()) + list(second_half_counts.keys())):
            first_count = first_half_counts.get(emotion, 0) / len(first_half) if first_half else 0
            second_count = second_half_counts.get(emotion, 0) / len(second_half) if second_half else 0
            
            if second_count > first_count:
                trend = "increasing"
            elif second_count < first_count:
                trend = "decreasing"
            else:
                trend = "stable"
                
            trends[emotion] = {
                "percentage": emotion_percentages.get(emotion, 0),
                "trend": trend,
                "change": second_count - first_count
            }
            
        return trends


class AdvancedEmotionDetector:
    """Enhanced emotion detection with advanced features"""
    
    def __init__(self, model_path=None, confidence_threshold=CONFIDENCE_THRESHOLD):
        """Initialize the advanced emotion detector"""
        self.confidence_threshold = confidence_threshold
        
        # Initialize emotion recognition model
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from {model_path}")
                self.emotion_model = EmotionRecognitionModel(model_path)
            else:
                logger.info("Initializing default emotion model")
                self.emotion_model = EmotionRecognitionModel()
                
            self.emotion_labels = self.emotion_model.emotion_labels
            logger.info(f"Model initialized with labels: {self.emotion_labels}")
        except Exception as e:
            logger.error(f"Failed to initialize emotion model: {e}")
            raise
            
        # Initialize face tracker
        self.face_tracker = FaceTracker()
        
        # Initialize emotion history
        self.emotion_history = EmotionHistory()
        
        # Visualization data
        self.visualization_queue = queue.Queue()
        self.vis_running = False
        self.vis_thread = None
        
    def detect_and_analyze_emotions(self, frame):
        """
        Detect faces and analyze emotions in a single frame
        
        Args:
            frame: Image as numpy array (BGR format from OpenCV)
            
        Returns:
            processed_frame: Frame with annotations
            results: Dictionary with detection and analysis results
        """
        # Convert to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            return frame, {"num_faces": 0, "faces": []}
            
        # Get face encodings for tracking
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Update face tracker
        face_ids = self.face_tracker.update(face_encodings, face_locations)
        
        # Process each face
        timestamp = time.time()
        results = {"num_faces": len(face_locations), "faces": []}
        
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            # Get face ID from tracker
            face_id = face_ids.get(i)
            
            if face_id is None:
                continue
                
            # Extract face
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Skip if face is too small
            if face_image.shape[0] < 20 or face_image.shape[1] < 20:
                continue
                
            # Convert to grayscale and resize for emotion model
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (48, 48))
            normalized_face = resized_face / 255.0
            model_input = normalized_face.reshape(1, 48, 48, 1)
            
            # Get emotion predictions
            predictions = self.emotion_model.model.predict(model_input, verbose=0)[0]
            
            # Get top prediction and confidence
            top_emotion_idx = np.argmax(predictions)
            top_confidence = predictions[top_emotion_idx]
            top_emotion = self.emotion_labels[top_emotion_idx]
            
            # Add to history
            self.emotion_history.add_emotion(face_id, timestamp, top_emotion, top_confidence)
            
            # Get emotion trend analysis
            emotion_trend = self.emotion_history.get_emotion_trend(face_id)
            
            # Get consistent emotion (reduces flickering)
            stable_emotion, stable_confidence = self.emotion_history.get_dominant_emotion(face_id)
            display_emotion = stable_emotion if stable_emotion else top_emotion
            
            # Confidence scoring
            confidence_score = {
                "raw_confidence": float(top_confidence),
                "stability": float(stable_confidence) if stable_confidence else 0.0,
                "is_reliable": float(top_confidence) >= self.confidence_threshold
            }
            
            # Draw bounding box and emotion
            color = EMOTION_COLORS.get(display_emotion, (255, 255, 255))
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw ID and emotion
            label = f"ID: {face_id} | {display_emotion.upper()} ({int(top_confidence * 100)}%)"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Store results
            face_result = {
                "face_id": face_id,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "emotions": {label: float(score) for label, score in zip(self.emotion_labels, predictions)},
                "top_emotion": top_emotion,
                "stable_emotion": stable_emotion,
                "confidence": confidence_score,
                "trend": emotion_trend
            }
            
            results["faces"].append(face_result)
            
        # Add to visualization queue
        if self.vis_running:
            try:
                self.visualization_queue.put({
                    "timestamp": timestamp,
                    "results": results
                }, block=False)
            except queue.Full:
                pass
                
        return frame, results
        
    def process_image(self, image_path, display=True, output_file=None):
        """
        Process a single image for emotion detection
        
        Args:
            image_path: Path to the input image file
            display: Whether to display the results
            output_file: Path to save the processed image
            
        Returns:
            results: Dictionary with detection and analysis results
        """
        # Read the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
                
            logger.info(f"Processing image: {image_path}")
            
            # Process the image
            processed_image, results = self.detect_and_analyze_emotions(image)
            
            # Save the processed image if requested
            if output_file:
                logger.info(f"Saving processed image to {output_file}")
                cv2.imwrite(output_file, processed_image)
                
            # Display the image if requested
            if display:
                cv2.imshow('Advanced Emotion Recognition', processed_image)
                logger.info("Displaying image. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None
    
    def process_webcam(self, camera_id=0, display=True, output_file=None):
        """
        Process webcam feed for real-time emotion detection
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the video feed
            output_file: Path to save the processed video
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Set up video writer if output file specified
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        
        # Start visualization thread if display is True
        if display:
            self.start_visualization()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to grab frame from camera")
                    break
                    
                # Process frame
                start_time = time.time()
                processed_frame, results = self.detect_and_analyze_emotions(frame)
                
                # Calculate FPS
                end_time = time.time()
                fps_actual = 1 / (end_time - start_time)
                
                # Display FPS on frame
                cv2.putText(processed_frame, f"FPS: {fps_actual:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Write to output file if specified
                if output_file:
                    out.write(processed_frame)
                
                # Show frame if display is True
                if display:
                    cv2.imshow('Advanced Emotion Recognition', processed_frame)
                    
                    # Exit on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            # Clean up
            cap.release()
            if output_file:
                out.release()
            if display:
                cv2.destroyAllWindows()
                self.stop_visualization()
    
    def start_visualization(self):
        """Start real-time visualization thread"""
        if self.vis_running:
            return
            
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self._visualization_thread)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
    def stop_visualization(self):
        """Stop visualization thread"""
        self.vis_running = False
        if self.vis_thread:
            self.vis_thread.join(timeout=1.0)
            self.vis_thread = None
            
    def _visualization_thread(self):
        """Thread for real-time emotion visualization"""
        # Set up the visualization
        plt.ion()  # Interactive mode
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title('Emotion Trends')
        
        # Emotion data for each face
        face_data = {}
        
        # Main visualization loop
        while self.vis_running:
            try:
                # Get latest data
                data = self.visualization_queue.get(timeout=0.1)
                timestamp = data["timestamp"]
                results = data["results"]
                
                # Clear figure
                fig.clear()
                
                # No faces detected
                if results["num_faces"] == 0:
                    plt.figtext(0.5, 0.5, "No faces detected", ha="center", va="center", fontsize=14)
                    plt.pause(0.01)
                    continue
                
                # Create subplots based on number of faces
                num_faces = min(4, results["num_faces"])  # Limit to 4 faces for visualization
                
                for i, face_result in enumerate(results["faces"][:num_faces]):
                    face_id = face_result["face_id"]
                    
                    # Create or update face data
                    if face_id not in face_data:
                        face_data[face_id] = {
                            "timestamps": [],
                            "emotions": {emotion: [] for emotion in self.emotion_labels}
                        }
                    
                    # Update face data
                    face_data[face_id]["timestamps"].append(timestamp)
                    
                    # Keep only recent data (last 100 points)
                    if len(face_data[face_id]["timestamps"]) > 100:
                        face_data[face_id]["timestamps"] = face_data[face_id]["timestamps"][-100:]
                    
                    # Update emotion values
                    for emotion in self.emotion_labels:
                        face_data[face_id]["emotions"][emotion].append(
                            face_result["emotions"].get(emotion, 0)
                        )
                        
                        # Keep only recent data
                        if len(face_data[face_id]["emotions"][emotion]) > 100:
                            face_data[face_id]["emotions"][emotion] = face_data[face_id]["emotions"][emotion][-100:]
                
                # Create subplot grid
                rows = (num_faces + 1) // 2
                cols = min(num_faces, 2)
                
                for i, face_result in enumerate(results["faces"][:num_faces]):
                    face_id = face_result["face_id"]
                    
                    # Create subplot
                    ax = fig.add_subplot(rows, cols, i+1)
                    ax.set_title(f"Face ID: {face_id} - {face_result['stable_emotion'] or face_result['top_emotion']}")
                    
                    # Plot emotion probabilities
                    x = np.arange(len(face_data[face_id]["timestamps"]))
                    for emotion in self.emotion_labels:
                        y = face_data[face_id]["emotions"][emotion]
                        if len(y) > 0:
                            ax.plot(x, y, label=emotion, color=tuple(c/255 for c in EMOTION_COLORS.get(emotion, (255, 255, 255))))
                    
                    # Add legend and labels
                    ax.legend(loc='upper left')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Probability')
                    ax.set_ylim(0, 1)
                    
                    # Add confidence threshold line
                    ax.axhline(y=self.confidence_threshold, color='r', linestyle='--', alpha=0.5)
                
                # Update display
                plt.tight_layout()
                plt.pause(0.01)
                
            except queue.Empty:
                plt.pause(0.01)
                continue
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                plt.pause(0.01)
                
        plt.ioff()
        plt.close(fig)
    
    def generate_emotion_report(self, face_id=None):
        """
        Generate a comprehensive emotion analysis report
        
        Args:
            face_id: Specific face ID to analyze, or None for all faces
            
        Returns:
            report: Dictionary with analysis results
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "faces": {}
        }
        
        # Filter face IDs if specified
        face_ids = [face_id] if face_id is not None else self.emotion_history.history.keys()
        
        for fid in face_ids:
            if fid not in self.emotion_history.history:
                continue
                
            history = self.emotion_history.history[fid]
            
            # Skip if not enough data
            if len(history["emotions"]) < 10:
                continue
                
            # Basic statistics
            emotion_counts = {}
            for emotion in history["emotions"]:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
                    
            total = len(history["emotions"])
            emotion_percentages = {emotion: count/total for emotion, count in emotion_counts.items()}
            
            # Get dominant emotion
            dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
            dominant_percentage = emotion_percentages[dominant_emotion]
            
            # Calculate confidence statistics
            confidence_values = list(history["confidences"])
            avg_confidence = sum(confidence_values) / len(confidence_values)
            max_confidence = max(confidence_values)
            min_confidence = min(confidence_values)
            
            # Calculate emotion stability
            changes = 0
            prev_emotion = None
            for emotion in history["emotions"]:
                if prev_emotion is not None and emotion != prev_emotion:
                    changes += 1
                prev_emotion = emotion
                
            stability = 1.0 - (changes / (total - 1)) if total > 1 else 1.0
            
            # Get trend analysis
            trends = self.emotion_history.get_emotion_trend(fid)
            
            # Store report for this face
            face_report = {
                "dominant_emotion": dominant_emotion,
                "dominant_percentage": dominant_percentage,
                "emotion_percentages": emotion_percentages,
                "confidence": {
                    "average": avg_confidence,
                    "max": max_confidence,
                    "min": min_confidence
                },
                "stability": stability,
                "trends": trends,
                "data_points": total
            }
            
            report["faces"][fid] = face_report
            
        return report


def main():
    """Main function to run the advanced emotion detector"""
    parser = argparse.ArgumentParser(description="Advanced Emotion Recognition System")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                      help="Path to pre-trained emotion model")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID")
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save processed video")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                      help="Confidence threshold for emotion detection")
    parser.add_argument("--no-display", action="store_true",
                      help="Disable video display")
    parser.add_argument("--image", type=str, default=None,
                      help="Path to image file for processing instead of webcam")
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = AdvancedEmotionDetector(
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # Process image or webcam
        if args.image:
            logger.info(f"Processing image: {args.image}")
            detector.process_image(
                image_path=args.image,
                display=not args.no_display,
                output_file=args.output
            )
        else:
            logger.info("Starting webcam processing")
            detector.process_webcam(
                camera_id=args.camera,
                display=not args.no_display,
                output_file=args.output
            )
        
    except KeyboardInterrupt:
        logger.info("Emotion detection stopped by user")
    except Exception as e:
        logger.error(f"Error during emotion detection: {e}")
        
        
if __name__ == "__main__":
    main()

