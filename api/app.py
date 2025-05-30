from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import logging
import time
import numpy as np
from PIL import Image
import io

# Add the parent directory to sys.path to allow imports from the model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.emotion_model import EmotionRecognitionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Facial Emotion Recognition API",
    description="API for recognizing emotions from facial expressions using TensorFlow",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize model
emotion_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the emotion recognition model when the application starts."""
    global emotion_model
    logger.info("Initializing emotion recognition model...")
    try:
        emotion_model = EmotionRecognitionModel()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/api/emotion/visual")
async def recognize_emotion(file: UploadFile = File(...)):
    """
    Recognize emotion from a facial image.
    
    Parameters:
    - file: Image file containing a face
    
    Returns:
    - JSON with emotion predictions
    """
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    if not emotion_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array for model processing
        image_np = np.array(image)
        
        # Process with model
        predictions = emotion_model.predict(image_np)
        
        return JSONResponse(content=predictions)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/api/emotion/visual/labels")
async def get_emotion_labels():
    """Get the list of emotion labels the model can recognize."""
    if not emotion_model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return {"labels": emotion_model.emotion_labels}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

