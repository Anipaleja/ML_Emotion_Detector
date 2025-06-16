# Multimodal Emotion Recognition System
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Java](https://img.shields.io/badge/Java-ED8B00?style=for-the-badge&logo=oracle&logoColor=white) ![Maven](https://img.shields.io/badge/Maven-C71A36?style=for-the-badge&logo=apachemaven&logoColor=white) ![Node.js](https://img.shields.io/badge/Node.js-339933?style=for-the-badge&logo=nodedotjs&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) ![WebRTC](https://img.shields.io/badge/WebRTC-333333?style=for-the-badge&logo=webrtc&logoColor=white) ![Microservices](https://img.shields.io/badge/Microservices-000000?style=for-the-badge&logo=microgenetics&logoColor=white)

A distributed neural network system that implements multimodal emotion recognition using multiple programming languages in a microservices architecture.




## Project Description

This project provides a comprehensive system for recognizing human emotions through multiple modalities:
- Facial expression analysis (visual)
- Voice tone analysis (audio)
- Combined multimodal prediction

The system is designed as a distributed microservices architecture where each component is built in a different programming language, demonstrating cross-language integration patterns and distributed AI system design.

## System Architecture

The system consists of the following components:

1. **Python Service**: TensorFlow-based facial emotion recognition
   - Processes video frames to detect faces
   - Analyzes facial expressions to predict emotions
   - Exposes predictions via REST API

2. **Java Service**: DSP-based audio emotion analysis
   - Processes audio streams to extract features using Digital Signal Processing
   - Predicts emotions from voice patterns
   - Exposes predictions via REST API

3. **Node.js Service**: Web interface and client-side processing
   - Captures real-time video and audio using WebRTC
   - Communicates with Python and Java services
   - Displays emotion predictions to the user

4. **Aggregator Service**: Combined emotion prediction
   - Collects predictions from both modalities
   - Applies fusion algorithms to generate final prediction
   - Provides unified API for clients

## Directory Structure

```
emotion_recognition/
├── python-service/         # Facial emotion recognition service
│   ├── model/              # TensorFlow model files
│   ├── api/                # REST API implementation
│   └── tests/              # Unit tests
│
├── java-service/           # Audio emotion analysis service
│   ├── src/                # Java source code
│   ├── lib/                # Dependencies
│   └── tests/              # Unit tests
│
├── nodejs-service/         # Web interface and client application
│   ├── public/             # Static assets
│   ├── src/                # JavaScript source code
│   ├── views/              # Web UI templates
│   └── tests/              # Unit tests
│
├── common/                 # Shared utilities and documentation
│   ├── docs/               # Project documentation
│   ├── schemas/            # Shared data schemas
│   └── utils/              # Common utility functions
│
├── docker/                 # Docker configuration
│   ├── docker-compose.yml  # Multi-container definition
│   └── */                  # Service-specific Dockerfiles
│
└── ci/                     # CI/CD configuration
```

## Prerequisites

- **Python 3.8+** with TensorFlow 2.x
- **Java 11+** with Maven
- **Node.js 16+** with npm
- **Docker** and Docker Compose
- Modern web browser with WebRTC support

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/emotion_recognition.git
cd emotion_recognition
```

### 2. Set Up Individual Services

#### Python Service

```bash
cd python-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Java Service

```bash
cd java-service
mvn install
```

#### Node.js Service

```bash
cd nodejs-service
npm install
```

### 3. Docker Setup (Optional)

To run the entire system using Docker:

```bash
cd docker
docker-compose up -d
```

## Usage Instructions

### Starting the Services Individually

1. **Python Service**:
   ```bash
   cd python-service
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python app.py
   ```

2. **Java Service**:
   ```bash
   cd java-service
   java -jar target/emotion-audio-1.0.jar
   ```

3. **Node.js Service**:
   ```bash
   cd nodejs-service
   npm start
   ```

### Accessing the Web Interface

Once all services are running, access the web interface at:
```
http://localhost:3000
```

### API Endpoints

- Facial Emotion API: `http://localhost:5000/api/emotion/visual`
- Audio Emotion API: `http://localhost:8080/api/emotion/audio`
- Combined Emotion API: `http://localhost:4000/api/emotion/combined`

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License
- See the LICENSE file for details.

## Acknowledgments

- FER+ dataset for facial emotion recognition
- RAVDESS dataset for audio emotion recognition
- TensorFlow team for their excellent deep learning framework

