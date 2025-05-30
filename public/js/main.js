// DOM Elements
const videoElement = document.getElementById('videoElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const snapshotButton = document.getElementById('snapshotButton');
const facialEmotion = document.getElementById('facialEmotion');
const voiceEmotion = document.getElementById('voiceEmotion');
const combinedEmotion = document.getElementById('combinedEmotion');
const facialEmotionBars = document.getElementById('facialEmotionBars');
const voiceEmotionBars = document.getElementById('voiceEmotionBars');
const logMessages = document.getElementById('logMessages');
const visualServiceStatus = document.getElementById('visualServiceStatus');
const audioServiceStatus = document.getElementById('audioServiceStatus');
const aggregatorServiceStatus = document.getElementById('aggregatorServiceStatus');

// Initialize variables
let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let websocket = null;
let videoProcessor = null;
let emotionChart = null;
let canvasContext = null;
let isProcessing = false;
let config = {
    frameRate: 5,
    audioSampleRate: 16000,
    processFrameInterval: 200 // ms between frame processing
};

// Emotion colors for visualization
const emotionColors = {
    angry: '#dc3545',
    disgust: '#6c757d',
    fear: '#6f42c1',
    happy: '#ffc107',
    sad: '#17a2b8',
    surprise: '#fd7e14',
    neutral: '#28a745'
};

// Initialize the application
function initialize() {
    // Set up WebSocket connection
    setupWebSocket();
    
    // Set up overlay canvas
    setupCanvas();
    
    // Create emotion chart
    createEmotionChart();
    
    // Set up button event listeners
    startButton.addEventListener('click', startCapture);
    stopButton.addEventListener('click', stopCapture);
    snapshotButton.addEventListener('click', takeSnapshot);
    
    // Check services status
    checkServicesStatus();
    
    // Add log message
    addLogMessage('System initialized');
}

// Set up WebSocket connection
function setupWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}`;
    
    websocket = new WebSocket(wsUrl);
    
    websocket.onopen = () => {
        addLogMessage('WebSocket connection established');
        setServiceStatus(visualServiceStatus, 'unknown');
        setServiceStatus(audioServiceStatus, 'unknown');
        setServiceStatus(aggregatorServiceStatus, 'unknown');
    };
    
    websocket.onclose = () => {
        addLogMessage('WebSocket connection closed');
        setServiceStatus(visualServiceStatus, 'offline');
        setServiceStatus(audioServiceStatus, 'offline');
        setServiceStatus(aggregatorServiceStatus, 'offline');
        
        // Attempt to reconnect after 5 seconds
        setTimeout(setupWebSocket, 5000);
    };
    
    websocket.onerror = (error) => {
        addLogMessage(`WebSocket error: ${error}`);
    };
    
    websocket.onmessage = (event) => {
        try {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
}

// Handle WebSocket messages
function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'config':
            // Update configuration
            config = { ...config, ...message.data };
            addLogMessage('Received configuration from server');
            break;
            
        case 'facial-emotion':
            // Update facial emotion display
            updateFacialEmotion(message.data);
            isProcessing = false;
            setServiceStatus(visualServiceStatus, 'online');
            break;
            
        case 'audio-emotion':
            // Update voice emotion display
            updateVoiceEmotion(message.data);
            setServiceStatus(audioServiceStatus, 'online');
            break;
            
        case 'combined-emotion':
            // Update combined emotion display
            updateCombinedEmotion(message.data);
            setServiceStatus(aggregatorServiceStatus, 'online');
            break;
            
        case 'error':
            addLogMessage(`Error: ${message.error}`);
            isProcessing = false;
            break;
            
        case 'pong':
            // Handle pong response (keep-alive)
            break;
            
        default:
            console.warn('Unknown message type:', message.type);
    }
}

// Check services status
function checkServicesStatus() {
    // Visual Service
    fetch('/api/emotion/visual/health')
        .then(response => {
            if (response.ok) {
                setServiceStatus(visualServiceStatus, 'online');
            } else {
                setServiceStatus(visualServiceStatus, 'offline');
            }
        })
        .catch(() => {
            setServiceStatus(visualServiceStatus, 'offline');
        });
    
    // Audio Service
    fetch('/api/emotion/audio/health')
        .then(response => {
            if (response.ok) {
                setServiceStatus(audioServiceStatus, 'online');
            } else {
                setServiceStatus(audioServiceStatus, 'offline');
            }
        })
        .catch(() => {
            setServiceStatus(audioServiceStatus, 'offline');
        });
    
    // Aggregator Service
    fetch('/api/emotion/combined/health')
        .then(response => {
            if (response.ok) {
                setServiceStatus(aggregatorServiceStatus, 'online');
            } else {
                setServiceStatus(aggregatorServiceStatus, 'offline');
            }
        })
        .catch(() => {
            setServiceStatus(aggregatorServiceStatus, 'offline');
        });
    
    // Check again in 30 seconds
    setTimeout(checkServicesStatus, 30000);
}

// Set service status indicator
function setServiceStatus(element, status) {
    element.className = 'status-indicator';
    if (status === 'online') {
        element.classList.add('status-online');
    } else if (status === 'offline') {
        element.classList.add('status-offline');
    } else if (status === 'processing') {
        element.classList.add('status-processing');
    }
}

// Set up canvas for overlaying face detection results
function setupCanvas() {
    canvasContext = overlayCanvas.getContext('2d');
    
    // Resize canvas to match video dimensions
    function resizeCanvas() {
        overlayCanvas.width = videoElement.offsetWidth;
        overlayCanvas.height = videoElement.offsetHeight;
    }
    
    // Resize initially and on window resize
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

// Create emotion chart
function createEmotionChart() {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    
    emotionChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'],
            datasets: [{
                label: 'Facial Emotion',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgb(255, 99, 132)',
                borderWidth: 1
            }, {
                label: 'Voice Emotion',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                borderWidth: 1
            }, {
                label: 'Combined',
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgb(75, 192, 192)',
                borderWidth: 2
            }]
        },
        options: {
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

// Start media capture
async function startCapture() {
    try {
        // Request access to camera and microphone
        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: true
        });
        
        // Set video source
        videoElement.srcObject = stream;
        
        // Set up audio recording
        setupAudioRecording(stream);
        
        // Start processing video frames
        startVideoProcessing();
        
        // Update UI
        startButton.disabled = true;
        stopButton.disabled = false;
        
        addLogMessage('Media capture started');
    } catch (error) {
        addLogMessage(`Error starting media capture: ${error.message}`);
        console.error('Error starting media capture:', error);
    }
}

// Set up audio recording
function setupAudioRecording(stream) {
    const audioContext = new AudioContext();
    const audioSource = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    
    audioSource.connect(processor);
    processor.connect(audioContext.destination);
    
    processor.onaudioprocess = (e) => {
        // Process audio data
        const inputData = e.inputBuffer.getChannelData(0);
        processAudioData(inputData);
    };
    
    // Also set up MediaRecorder for sending chunks
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
            
            // When we have enough data, send it
            if (audioChunks.length >= 10) {
                sendAudioData();
            }
        }
    };
    
    mediaRecorder.start(1000); // Collect data every second
}

// Process audio data
function processAudioData(audioData) {
    // This function would process audio in real-time
    // For this demo, we'll use the MediaRecorder approach instead
}

// Send audio data to server
function sendAudioData() {
    if (!websocket || audioChunks.length === 0) return;
    
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    audioChunks = []; // Clear the chunks
    
    // Convert to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        
        // Send via WebSocket
        websocket.send(JSON.stringify({
            type: 'audio-data',
            audio: base64Audio
        }));
    };
}

// Start processing video frames
function startVideoProcessing() {
    videoProcessor = setInterval(() => {
        processVideoFrame();
    }, config.processFrameInterval);
}

// Process current video frame
function processVideoFrame() {
    if (!videoElement.srcObject || isProcessing) return;
    
    isProcessing = true;
    setServiceStatus(visualServiceStatus, 'processing');
    
    // Create a temporary canvas to capture the frame
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoElement.videoWidth;
    tempCanvas.height = videoElement.videoHeight;
    
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Convert to base64
    const base64Frame = tempCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
    
    // Send to server via WebSocket
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'video-frame',
            frame: base64Frame
        }));
    } else {
        isProcessing = false;
    }
}

// Stop media capture
function stopCapture() {
    // Stop video processing
    if (videoProcessor) {
        clearInterval(videoProcessor);
        videoProcessor = null;
    }
    
    // Stop media recorder
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    // Stop all media tracks
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
        stream = null;
    }
    
    // Update UI
    startButton.disabled = false;
    stopButton.disabled = true;
    
    // Clear canvas
    if (canvasContext) {
        canvasContext.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    
    addLogMessage('Media capture stopped');
}

// Take a snapshot and process it
function takeSnapshot() {
    if (!videoElement.srcObject) {
        addLogMessage('No video stream available');
        return;
    }
    
    // Process a single frame
    processVideoFrame();
    
    addLogMessage('Snapshot taken and sent for processing');
}

// Update facial emotion display
function updateFacialEmotion(data) {
    // Clear previous content
    facialEmotionBars.innerHTML = '';
    
    // If no faces detected
    if (data.error || data.num_faces === 0) {
        facialEmotion.textContent = 'No face detected';
        return;
    }
    
    // Get the first face
    const face = data.predictions[0];
    const dominantEmotion = face.dominant_emotion;
    
    // Display dominant emotion
    facialEmotion.textContent = capitalizeFirstLetter(dominantEmotion);
    
    // Create progress bars for each emotion
    for (const [emotion, score] of Object.entries(face.emotions)) {
        createEmotionProgressBar(facialEmotionBars, emotion, score, emotionColors[emotion]);
    }
    
    // Draw face rectangle on canvas
    drawFaceRectangle(face.location);
    
    // Update chart
    updateEmotionChart('facial', face.emotions);
}

// Update voice emotion display
function updateVoiceEmotion(data) {
    // Clear previous content
    voiceEmotionBars.innerHTML = '';
    
    // If error or no data
    if (data.error || !data.emotions) {
        voiceEmotion.textContent = 'No audio data';
        return;
    }
    
    // Display dominant emotion
    const dominantEmotion = data.dominant_emotion;
    voiceEmotion.textContent = capitalizeFirstLetter(dominantEmotion);
    
    // Create progress bars for each emotion
    for (const [emotion, score] of Object.entries(data.emotions)) {
        createEmotionProgressBar(voiceEmotionBars, emotion, score, emotionColors[emotion]);
    }
    
    // Update chart
    updateEmotionChart('voice', data.emotions);
}

// Update combined emotion display
function updateCombinedEmotion(data) {
    // If error or no data
    if (data.error || !data.combined_emotion) {
        combinedEmotion.textContent = 'Not available';
        return;
    }
    
    // Display combined emotion
    combinedEmotion.textContent = capitalizeFirstLetter(data.combined_emotion);
    
    // Update chart
    updateEmotionChart('combined', data.emotion_scores);
}

// Update emotion chart
function updateEmotionChart(type, emotions) {
    if (!emotionChart) return;
    
    const datasetIndex = type === 'facial' ? 0 : (type === 'voice' ? 1 : 2);
    const labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
    
    // Update data for the specified dataset
    emotionChart.data.datasets[datasetIndex].data = labels.map(label => 
        emotions[label] !== undefined ? emotions[label] : 0
    );
    
    emotionChart.update();
}

// Create a progress bar for emotion visualization
function createEmotionProgressBar(container, emotion, score, color) {
    const percent = Math.round(score * 100);
    
    const wrapper = document.createElement('div');
    wrapper.className = 'mb-2';
    
    const label = document.createElement('div');
    label.className = 'd-flex justify-content-between';
    label.innerHTML = `
        <span>${capitalizeFirstLetter(emotion)}</span>
        <span>${percent}%</span>
    `;
    
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.width = `${percent}%`;
    progressBar.style.backgroundColor = color;
    progressBar.setAttribute('role', 'progressbar');
    progressBar.setAttribute('aria-valuenow', percent);
    progressBar.setAttribute('aria-valuemin', 0);
    progressBar.setAttribute('aria-valuemax', 100);
    
    progressContainer.appendChild(progressBar);
    wrapper.appendChild(label);
    wrapper.appendChild(progressContainer);
    container.appendChild(wrapper);
}

// Draw face rectangle on canvas
function drawFaceRectangle(location) {
    if (!canvasContext || !location) return;
    
    // Clear previous drawings
    can

