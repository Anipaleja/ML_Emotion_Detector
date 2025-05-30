package com.emotion.audio;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main application class for the Audio Emotion Recognition service.
 * This Spring Boot application provides REST endpoints for audio-based
 * emotion recognition using Digital Signal Processing and Deep Learning.
 */
@SpringBootApplication
public class AudioEmotionApplication {

    private static final Logger logger = LoggerFactory.getLogger(AudioEmotionApplication.class);

    public static void main(String[] args) {
        logger.info("Starting Audio Emotion Recognition Service");
        SpringApplication.run(AudioEmotionApplication.class, args);
    }

    /**
     * Configure CORS to allow cross-origin requests from the web client.
     * This is necessary for the microservices architecture where the web client
     * and audio service may be running on different domains/ports.
     */
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/**")
                        .allowedOrigins("*")
                        .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                        .allowedHeaders("*");
            }
        };
    }
}

