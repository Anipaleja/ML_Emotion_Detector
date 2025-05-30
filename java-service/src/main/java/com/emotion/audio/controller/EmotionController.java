package com.emotion.audio.controller;

import com.emotion.audio.model.AudioData;
import com.emotion.audio.model.EmotionResult;
import com.emotion.audio.service.EmotionRecognitionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Base64;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

/**
 * REST Controller for the audio emotion recognition endpoints.
 * Provides API endpoints for health checks and emotion recognition.
 */
@RestController
@RequestMapping("/api/emotion/audio")
public class EmotionController {

    private static final Logger logger = LoggerFactory.getLogger(EmotionController.class);

    private final EmotionRecognitionService emotionService;

    @Autowired
    public EmotionController(EmotionRecognitionService emotionService) {
        this.emotionService = emotionService;
    }

    /**
     * Health check endpoint to verify the service is running.
     * 
     * @return A status response with timestamp
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "healthy");
        response.put("timestamp", new Date().toInstant().toString());
        response.put("service", "java-audio-emotion-service");
        
        return ResponseEntity.ok(response);
    }

    /**
     * Endpoint for processing an audio file upload and recognizing emotions.
     * 
     * @param audioFile The audio file to analyze
     * @return Emotion recognition results
     */
    @PostMapping(value = "", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<EmotionResult> recognizeEmotionFromFile(
            @RequestParam("file") MultipartFile audioFile) {
        
        logger.info("Received audio file for emotion recognition: {}", audioFile.getOriginalFilename());
        
        try {
            byte[] audioData = audioFile.getBytes();
            EmotionResult result = emotionService.recognizeEmotion(audioData);
            return ResponseEntity.ok(result);
        } catch (IOException e) {
            logger.error("Error processing audio file", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }

    /**
     * Endpoint for processing base64-encoded audio data and recognizing emotions.
     * This is used by the web client which sends audio data via WebSockets.
     * 
     * @param audioData Object containing base64-encoded audio data
     * @return Emotion recognition results
     */
    @PostMapping(value = "", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<EmotionResult> recognizeEmotionFromBase64(
            @RequestBody AudioData audioData) {
        
        logger.info("Received base64 audio data for emotion recognition");
        
        try {
            if (audioData.getAudio() == null || audioData.getAudio().isEmpty()) {
                return ResponseEntity.badRequest().body(null);
            }
            
            byte[] decodedAudio = Base64.getDecoder().decode(audioData.getAudio());
            EmotionResult result = emotionService.recognizeEmotion(decodedAudio);
            return ResponseEntity.ok(result);
        } catch (IllegalArgumentException e) {
            logger.error("Error decoding base64 audio data", e);
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        } catch (Exception e) {
            logger.error("Error processing audio data", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }
    
    /**
     * Returns information about the available audio emotion labels.
     * 
     * @return A list of emotion labels the system can recognize
     */
    @GetMapping("/labels")
    public ResponseEntity<Map<String, Object>> getEmotionLabels() {
        Map<String, Object> response = new HashMap<>();
        response.put("labels", emotionService.getEmotionLabels());
        
        return ResponseEntity.ok(response);
    }
}

