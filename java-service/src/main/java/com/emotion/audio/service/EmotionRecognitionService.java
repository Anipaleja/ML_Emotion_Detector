package com.emotion.audio.service;

import com.emotion.audio.model.EmotionResult;
import com.github.jlibrosa.audio.JLibrosa;
import com.github.jlibrosa.audio.exception.FileFormatNotSupportedException;
import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service for audio-based emotion recognition using digital signal processing
 * and deep learning techniques.
 */
@Service
public class EmotionRecognitionService {

    private static final Logger logger = LoggerFactory.getLogger(EmotionRecognitionService.class);
    
    private static final int SAMPLE_RATE = 16000;
    private static final int FFT_SIZE = 512;
    private static final int HOP_LENGTH = 256;
    private static final int N_MELS = 40;
    private static final int NUM_MFCC = 20;
    
    private static final List<String> EMOTION_LABELS = Arrays.asList(
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral");
    
    private MultiLayerNetwork model;
    private JLibrosa jLibrosa;
    
    /**
     * Initialize the service, loading the model and setting up JLibrosa.
     */
    @PostConstruct
    public void init() {
        try {
            // Initialize JLibrosa for audio processing
            jLibrosa = new JLibrosa();
            
            // Load pre-trained model if it exists
            loadModel();
            
            logger.info("EmotionRecognitionService initialized successfully");
        } catch (Exception e) {
            logger.error("Error initializing EmotionRecognitionService", e);
        }
    }
    
    /**
     * Load the pre-trained emotion recognition model.
     * For demonstration purposes, we'll create a dummy model if none exists.
     */
    private void loadModel() {
        try {
            // Try to load a pre-trained model from the classpath resources
            File modelFile = new ClassPathResource("models/emotion_model.zip").getFile();
            model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            logger.info("Loaded pre-trained emotion recognition model");
        } catch (IOException e) {
            logger.warn("Pre-trained model not found, creating a simple model for demonstration");
            createDummyModel();
        }
    }
    
    /**
     * Create a simple dummy model for demonstration purposes.
     * In a real implementation, this would be replaced with a properly trained model.
     */
    private void createDummyModel() {
        // This is a placeholder for demonstration. In a real system,
        // you would train a proper model using DL4J or load a pre-trained one.
        // For simplicity, we'll create a dummy implementation that returns random predictions.
        model = null;
        logger.info("Created dummy model for demonstration");
    }
    
    /**
     * Recognize emotions from audio data.
     * 
     * @param audioData Raw audio data bytes
     * @return Emotion recognition results
     */
    public EmotionResult recognizeEmotion(byte[] audioData) {
        try {
            // For demo purposes, if no model is loaded, return random predictions
            if (model == null) {
                return generateDummyPredictions();
            }
            
            // Extract audio features (MFCC)
            float[][] mfccFeatures = extractMFCC(audioData);
            
            // Convert features to the format expected by the model
            INDArray features = prepareFeatures(mfccFeatures);
            
            // Get model predictions
            INDArray output = model.output(features);
            
            // Process the model output
            return processModelOutput(output);
            
        } catch (Exception e) {
            logger.error("Error recognizing emotion from audio", e);
            return generateDummyPredictions(); // Fallback to dummy predictions on error
        }
    }
    
    /**
     * Extract MFCC features from audio data.
     * 
     * @param audioData Raw audio data bytes
     * @return MFCC features matrix
     * @throws IOException If audio processing fails
     */
    private float[][] extractMFCC(byte[] audioData) throws IOException {
        try {
            // Save audio data to a temporary file
            Path tempFile = Files.createTempFile("audio_", ".wav");
            FileUtils.writeByteArrayToFile(tempFile.toFile(), audioData);
            
            // Extract MFCC features using JLibrosa
            float[][] mfccFeatures = jLibrosa.getMFCC(tempFile.toString(), SAMPLE_RATE, NUM_MFCC);
            
            // Clean up temporary file
            Files.delete(tempFile);
            
            return mfccFeatures;
        } catch (FileFormatNotSupportedException e) {
            logger.error("Audio file format not supported", e);
            throw new IOException("Audio file format not supported", e);
        }
    }
    
    /**
     * Prepare features for the model.
     * 
     * @param mfccFeatures MFCC features matrix
     * @return Features in the format expected by the model
     */
    private INDArray prepareFeatures(float[][] mfccFeatures) {
        // Convert 2D feature matrix to the format expected by the model
        // This would depend on your specific model architecture
        
        // For demonstration, we'll create a simple feature vector
        // by averaging the MFCC coefficients over time
        float[] meanFeatures = new float[NUM_MFCC];
        
        for (int i = 0; i < NUM_MFCC; i++) {
            float sum = 0;
            for (int j = 0; j < mfccFeatures.length; j++) {
                sum += mfccFeatures[j][i];
            }
            meanFeatures[i] = sum / mfccFeatures.length;
        }
        
        // Create an ND4J array and reshape as needed for the model
        INDArray features = Nd4j.create(meanFeatures);
        
        return features.reshape(1, NUM_MFCC);
    }
    
    /**
     * Process model output to generate emotion predictions.
     * 
     * @param output Model output
     * @return Processed emotion recognition results
     */
    private EmotionResult processModelOutput(INDArray output) {
        // Get the raw probabilities from the model output
        float[] probabilities = output.toFloatVector();
        
        // Create a map of emotion probabilities
        Map<String, Float> emotions = new HashMap<>();
        for (int i = 0; i < EMOTION_LABELS.size(); i++) {
            emotions.put(EMOTION_LABELS.get(i), probabilities[i]);
        }
        
        // Find the dominant emotion
        String dominantEmotion = EMOTION_LABELS.get(0);
        float maxProb = probabilities[0];
        
        for (int i = 1; i < EMOTION_LABELS.size(); i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                dominantEmotion = EMOTION_LABELS.get(i);
            }
        }
        
        // Create and return the result
        return new EmotionResult(emotions, dominantEmotion);
    }
    
    /**
     * Generate dummy predictions for demonstration purposes.
     * In a real system, this would be replaced with actual model predictions.
     * 
     * @return Random emotion predictions
     */
    private EmotionResult generateDummyPredictions() {
        // Generate random probabilities
        float[] probabilities = new float[EMOTION_LABELS.size()];
        float sum = 0;
        
        // Generate random values
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (float) Math.random();
            sum += probabilities[i];
        }
        
        // Normalize to sum to 1
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        
        // Create a map of emotion probabilities
        Map<String, Float> emotions = new HashMap<>();
        for (int i = 0; i < EMOTION_LABELS.size(); i++) {
            emotions.put(EMOTION_LABELS.get(i), probabilities[i]);
        }
        
        // Find the dominant emotion
        String dominantEmotion = EMOTION_LABELS.get(0);
        float maxProb = probabilities[0];
        
        for (int i = 1; i < EMOTION_LABELS.size(); i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                dominantEmotion = EMOTION_LABELS.get(i);
            }
        }
        
        // Return the result
        return new EmotionResult(emotions, dominantEmotion);
    }
    
    /**
     * Get the list of emotion labels the model can recognize.
     * 
     * @return List of emotion labels
     */
    public List<String> getEmotionLabels() {
        return EMOTION_LABELS;
    }
}

