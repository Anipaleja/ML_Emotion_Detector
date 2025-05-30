#!/usr/bin/env python
import os
import sys
import logging
import requests
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import zipfile
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define model path
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "saved_model")
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model")

# Define emotion labels (must match those in emotion_model.py)
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def download_file(url, destination):
    """
    Download a file from a URL with progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size from headers
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB blocks
        
        logger.info(f"Downloading from {url} to {destination}")
        with open(destination, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
                
        logger.info("Download completed successfully.")
        return True
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False

def create_new_model():
    """
    Create a model with the same architecture as in emotion_model.py
    """
    img_height = 48
    img_width = 48
    
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
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
    model.add(Dense(len(EMOTION_LABELS), activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    logger.info("Created new model with emotion recognition architecture")
    return model

def download_pretrained_model():
    """
    Download a pre-trained emotion recognition model
    """
    # URL to a pre-trained FER model (this is an example URL)
    # In a real-world scenario, you would point to an actual hosted model
    model_url = "https://github.com/priya-dwivedi/face_and_emotion_detection/raw/master/emotion_detector_models/model_v6_23.hdf5"
    
    try:
        # Create a temporary directory for downloading
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_model_path = os.path.join(temp_dir, "temp_model.h5")
            
            # Download the pre-trained model
            if not download_file(model_url, temp_model_path):
                logger.error("Failed to download pre-trained model")
                return None
            
            try:
                # Try to load the downloaded model
                logger.info("Loading downloaded model")
                model = load_model(temp_model_path)
                
                # Verify model structure (check output shape)
                output_shape = model.output_shape
                expected_shape = (None, len(EMOTION_LABELS))
                
                if output_shape[-1] != expected_shape[-1]:
                    logger.warning(f"Model output shape {output_shape} doesn't match expected shape {expected_shape}")
                    logger.warning("Creating a new model with correct architecture instead")
                    return create_new_model()
                
                logger.info(f"Successfully loaded pre-trained model with output shape: {output_shape}")
                return model
            
            except Exception as e:
                logger.error(f"Error loading downloaded model: {e}")
                logger.warning("Creating a new model with correct architecture instead")
                return create_new_model()
    
    except Exception as e:
        logger.error(f"Error during model download process: {e}")
        return None

def main():
    """
    Main function to download and save the emotion recognition model
    """
    # Make sure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        logger.info(f"Model already exists at {MODEL_PATH}")
        choice = input("Model already exists. Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            logger.info("Download cancelled by user")
            return
    
    # Download pre-trained model
    logger.info("Downloading pre-trained emotion recognition model...")
    model = download_pretrained_model()
    
    if model is None:
        logger.error("Failed to obtain a model. Exiting.")
        sys.exit(1)
    
    # Save the model
    try:
        logger.info(f"Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)
        logger.info("Model saved successfully!")
        logger.info(f"Model architecture summary:")
        model.summary(print_fn=logger.info)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

