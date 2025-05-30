import sys
import os
import unittest
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.emotion_model import EmotionRecognitionModel

class TestEmotionRecognitionModel(unittest.TestCase):
    """Test cases for the EmotionRecognitionModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model for testing
        with patch('tensorflow.keras.models.Sequential') as mock_sequential:
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            self.model = EmotionRecognitionModel()
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.emotion_labels), 7)  # 7 emotion classes
        self.assertEqual(self.model.img_height, 48)
        self.assertEqual(self.model.img_width, 48)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Process the image
        processed = self.model._preprocess_image(test_image)
        
        # Check output shape and type
        self.assertEqual(processed.shape, (1, 48, 48, 1))
        self.assertEqual(processed.dtype, np.float32)
        self.assertTrue(0 <= np.max(processed) <= 1.0)  # Check normalization
    
    def test_detect_faces_no_face(self):
        """Test face detection with no faces."""
        # Create an empty image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face_recognition.face_locations to return empty list
        with patch('face_recognition.face_locations', return_value=[]):
            face_locations = self.model._detect_faces(test_image)
            self.assertEqual(len(face_locations), 0)
    
    def test_detect_faces_with_face(self):
        """Test face detection with a face."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face_recognition.face_locations to return a face location
        mock_location = [(10, 50, 60, 20)]  # top, right, bottom, left
        with patch('face_recognition.face_locations', return_value=mock_location):
            face_locations = self.model._detect_faces(test_image)
            self.assertEqual(len(face_locations), 1)
            self.assertEqual(face_locations[0], mock_location[0])
    
    def test_extract_face(self):
        """Test face extraction from an image."""
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        # Create a test face location
        face_location = (10, 50, 60, 20)  # top, right, bottom, left
        
        # Extract face
        face_image = self.model._extract_face(test_image, face_location)
        
        # Check dimensions
        self.assertEqual(face_image.shape, (50, 30, 3))  # height=bottom-top, width=right-left
    
    def test_predict_no_faces(self):
        """Test prediction when no faces are detected."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face detection to return no faces
        with patch.object(self.model, '_detect_faces', return_value=[]):
            result = self.model.predict(test_image)
            
            # Check result structure
            self.assertIn('error', result)
            self.assertIn('predictions', result)
            self.assertEqual(len(result['predictions']), 0)
    
    def test_predict_with_face(self):
        """Test prediction with a detected face."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock face detection to return a face
        mock_location = (10, 50, 60, 20)
        with patch.object(self.model, '_detect_faces', return_value=[mock_location]):
            # Mock face extraction
            with patch.object(self.model, '_extract_face', return_value=np.zeros((50, 30, 3))):
                # Mock preprocessing
                with patch.object(self.model, '_preprocess_image', return_value=np.zeros((1, 48, 48, 1))):
                    # Mock model prediction
                    mock_prediction = np.array([0.1, 0.05, 0.05, 0.6, 0.05, 0.05, 0.1])  # Happy is dominant
                    with patch.object(self.model.model, 'predict', return_value=np.array([mock_prediction])):
                        result = self.model.predict(test_image)
                        
                        # Check result structure
                        self.assertEqual(result['num_faces'], 1)
                        self.assertEqual(len(result['predictions']), 1)
                        self.assertEqual(result['predictions'][0]['dominant_emotion'], 'happy')
                        self.assertAlmostEqual(result['predictions'][0]['emotions']['happy'], 0.6)

# For running with pytest
if __name__ == "__main__":
    pytest.main(["-v", __file__])

