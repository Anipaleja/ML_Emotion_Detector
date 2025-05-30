import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import logging
import cv2
import face_recognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class EmotionRecognitionModel:
    """
    A class to handle facial emotion recognition using TensorFlow.
    
    This model can:
    1. Create and train a CNN model for emotion recognition
    2. Load a pre-trained model
    3. Detect faces in images
    4. Predict emotions from facial images
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the emotion recognition model.
        
        Args:
            model_path (str, optional): Path to a pre-trained model. If None, 
                                        a new model will be created.
        """
        # Define emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Define image parameters
        self.img_height = 48
        self.img_width = 48
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading pre-trained model from {model_path}")
            self.model = self._load_model(model_path)
        else:
            logger.info("Creating new emotion recognition model")
            self.model = self._create_model()
    
    def _create_model(self):
        """
        Create a CNN model for emotion recognition.
        
        Returns:
            model: A TensorFlow/Keras model
        """
        model = Sequential()
        
        # First convolutional layer
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional layer
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.emotion_labels), activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        logger.info("Model created successfully")
        return model
    
    def _load_model(self, model_path):
        """
        Load a pre-trained model from disk.
        
        Args:
            model_path (str): Path to the model file
            
        Returns:
            model: The loaded TensorFlow/Keras model
        """
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self, train_data, validation_data, epochs=50, batch_size=64):
        """
        Train the emotion recognition model.
        
        Args:
            train_data: Training data generator or tuple (x_train, y_train)
            validation_data: Validation data generator or tuple (x_val, y_val)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            history: Training history
        """
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        logger.info("Model training completed")
        return history
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _preprocess_image(self, image):
        """
        Preprocess an image for the model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            processed_image: Preprocessed image ready for the model
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to expected dimensions
        resized = cv2.resize(gray, (self.img_width, self.img_height))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Reshape for model input
        processed = normalized.reshape(1, self.img_height, self.img_width, 1)
        
        return processed
    
    def _detect_faces(self, image):
        """
        Detect faces in an image.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            face_locations: List of face bounding boxes
        """
        # Convert BGR to RGB if needed (face_recognition expects RGB)
        if len(image.shape) > 2 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        return face_locations
    
    def _extract_face(self, image, face_location):
        """
        Extract a face from an image based on its location.
        
        Args:
            image (numpy.ndarray): Input image
            face_location (tuple): Face location as (top, right, bottom, left)
            
        Returns:
            face_image: Extracted face image
        """
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        return face_image
    
    def predict(self, image):
        """
        Predict emotions from an image containing faces.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            predictions: Dictionary with emotion predictions
        """
        try:
            # Detect faces
            face_locations = self._detect_faces(image)
            
            if not face_locations:
                logger.warning("No faces detected in the image")
                return {"error": "No faces detected", "predictions": []}
            
            all_predictions = []
            
            # Process each detected face
            for i, face_location in enumerate(face_locations):
                # Extract face
                face_image = self._extract_face(image, face_location)
                
                # Preprocess face for model
                processed_face = self._preprocess_image(face_image)
                
                # Get model predictions
                prediction = self.model.predict(processed_face)[0]
                
                # Create prediction dictionary
                face_prediction = {
                    "face_id": i,
                    "location": {
                        "top": face_location[0],
                        "right": face_location[1],
                        "bottom": face_location[2],
                        "left": face_location[3]
                    },
                    "emotions": {label: float(score) for label, score in zip(self.emotion_labels, prediction)}
                }
                
                # Add dominant emotion
                dominant_emotion = self.emotion_labels[np.argmax(prediction)]
                face_prediction["dominant_emotion"] = dominant_emotion
                
                all_predictions.append(face_prediction)
            
            return {
                "num_faces": len(all_predictions),
                "predictions": all_predictions
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e), "predictions": []}


# For demonstration/testing purposes
if __name__ == "__main__":
    # Example usage of the model
    model = EmotionRecognitionModel()
    print("Model initialized for testing")

