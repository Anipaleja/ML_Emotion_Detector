package com.emotion.audio.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Model class to represent incoming audio data in base64 format.
 * Used for receiving audio data via REST API.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class AudioData {
    
    /**
     * Base64-encoded audio data.
     */
    private String audio;
}

