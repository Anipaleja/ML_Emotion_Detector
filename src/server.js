const express = require('express');
const http = require('http');
const path = require('path');
const WebSocket = require('ws');
const cors = require('cors');
const axios = require('axios');
const { createProxyMiddleware } = require('http-proxy-middleware');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Configuration
const PORT = process.env.PORT || 3000;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';
const JAVA_SERVICE_URL = process.env.JAVA_SERVICE_URL || 'http://localhost:8080';
const AGGREGATOR_SERVICE_URL = process.env.AGGREGATOR_SERVICE_URL || 'http://localhost:4000';

// Initialize Express app
const app = express();
const server = http.createServer(app);

// Create WebSocket server
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// Set up proxy for the Python service (facial emotion recognition)
app.use('/api/emotion/visual', createProxyMiddleware({
    target: PYTHON_SERVICE_URL,
    changeOrigin: true,
    pathRewrite: {
        '^/api/emotion/visual': '/api/emotion/visual'
    },
    onError: (err, req, res) => {
        console.error('Proxy error:', err);
        res.status(500).json({ error: 'Python service unavailable' });
    }
}));

// Set up proxy for the Java service (audio emotion recognition)
app.use('/api/emotion/audio', createProxyMiddleware({
    target: JAVA_SERVICE_URL,
    changeOrigin: true,
    pathRewrite: {
        '^/api/emotion/audio': '/api/emotion/audio'
    },
    onError: (err, req, res) => {
        console.error('Proxy error:', err);
        res.status(500).json({ error: 'Java service unavailable' });
    }
}));

// Set up proxy for the Aggregator service (combined emotion recognition)
app.use('/api/emotion/combined', createProxyMiddleware({
    target: AGGREGATOR_SERVICE_URL,
    changeOrigin: true,
    pathRewrite: {
        '^/api/emotion/combined': '/api/emotion/combined'
    },
    onError: (err, req, res) => {
        console.error('Proxy error:', err);
        res.status(500).json({ error: 'Aggregator service unavailable' });
    }
}));

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'nodejs-web-interface'
    });
});

// WebSocket connection handling
wss.on('connection', (ws) => {
    console.log('WebSocket client connected');

    // Handle incoming messages from clients
    ws.on('message', async (message) => {
        try {
            const data = JSON.parse(message);
            
            // Handle different message types
            switch (data.type) {
                case 'video-frame':
                    // Process video frame and forward to Python service
                    try {
                        // Convert base64 to blob and send to Python service
                        const response = await axios.post(`${PYTHON_SERVICE_URL}/api/emotion/visual`, {
                            image: data.frame
                        }, {
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        // Send the response back to the client
                        ws.send(JSON.stringify({
                            type: 'facial-emotion',
                            data: response.data
                        }));
                    } catch (error) {
                        console.error('Error processing video frame:', error);
                        ws.send(JSON.stringify({
                            type: 'error',
                            error: 'Failed to process video frame'
                        }));
                    }
                    break;
                    
                case 'audio-data':
                    // Process audio data and forward to Java service
                    try {
                        // Send audio data to Java service
                        const response = await axios.post(`${JAVA_SERVICE_URL}/api/emotion/audio`, {
                            audio: data.audio
                        }, {
                            headers: {
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        // Send the response back to the client
                        ws.send(JSON.stringify({
                            type: 'audio-emotion',
                            data: response.data
                        }));
                    } catch (error) {
                        console.error('Error processing audio data:', error);
                        ws.send(JSON.stringify({
                            type: 'error',
                            error: 'Failed to process audio data'
                        }));
                    }
                    break;
                
                case 'ping':
                    ws.send(JSON.stringify({ type: 'pong' }));
                    break;
                
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error processing message:', error);
        }
    });

    // Handle WebSocket connection close
    ws.on('close', () => {
        console.log('WebSocket client disconnected');
    });

    // Send initial configuration to the client
    ws.send(JSON.stringify({
        type: 'config',
        data: {
            frameRate: 5, // Frames per second to process
            audioSampleRate: 16000 // Audio sample rate
        }
    }));
});

// Start the server
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log(`Python Service URL: ${PYTHON_SERVICE_URL}`);
    console.log(`Java Service URL: ${JAVA_SERVICE_URL}`);
    console.log(`Aggregator Service URL: ${AGGREGATOR_SERVICE_URL}`);
});

// Handle server shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM signal received: closing HTTP server');
    server.close(() => {
        console.log('HTTP server closed');
    });
});

