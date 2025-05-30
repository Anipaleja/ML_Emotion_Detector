package com.emotion.audio.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

/**
 * Model class to represent emotion recognition results.
 * Contains a map of emotions with their probabilities and the dominant emotion.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class EmotionResult {
    
    /**
     * Map of emotions to their probability scores.
     * Each key is an emotion label (e.g., "happy", "sad", etc.)
     * and each value is a probability between 0 and 1.
     */
    private Map<String, Float> emotions;
    
    /**
     * The dominant emotion with the highest probability.
     */
    private String dominantEmotion;
}

