# PowerBot - Transmission Line Fault Detection System

An AI-powered system for detecting and diagnosing transmission line faults using machine learning and RAG (Retrieval-Augmented Generation) technology.

## 📋 Project Overview

PowerBot is an intelligent fault detection and diagnosis system designed for transmission line monitoring in power systems. The system leverages machine learning models to classify different types of electrical faults (LLLG, LLG, LG) based on voltage and current readings from three-phase power systems. Additionally, it integrates a RAG-based AI agent to provide detailed diagnostic information and recommended solutions.

### Problem Statement
Transmission line faults can cause significant power outages, equipment damage, and safety hazards. Early and accurate fault detection is crucial for maintaining grid reliability and minimizing downtime. This system provides real-time fault classification with actionable insights for power system operators.

## ✨ Key Features

- **Multi-Class Fault Classification**: Detects and classifies four fault types:
  - No Fault (Normal Operation)
  - LLLG Fault (Three-Phase-to-Ground)
  - LLG Fault (Double Line-to-Ground)
  - LG Fault (Single Line-to-Ground)

- **High Accuracy Predictions**: Machine learning pipeline with confidence scores for each prediction

- **AI-Powered Diagnostics**: RAG-based system providing detailed fault explanations and recommended solutions

- **RESTful API**: FastAPI-based backend for easy integration with monitoring systems

- **Interactive Web Interface**: User-friendly chatbot interface (PowerBot) for submitting readings and viewing diagnostics

- **Real-Time Analysis**: Instant fault detection and classification from voltage and current measurements

## 🛠️ Tech Stack

### Backend
- **Python 3.x**: Core programming language
- **FastAPI**: High-performance web framework for API development
- **Scikit-learn**: Machine learning model training and inference
- **Pandas**: Data manipulation and feature engineering
- **NumPy**: Numerical computations
- **Joblib**: Model serialization and loading
- **LangChain**: RAG implementation and LLM integration
- **Uvicorn**: ASGI server for FastAPI

### Frontend
- **HTML5/CSS3**: Web interface structure and styling
- **JavaScript (Vanilla)**: Client-side interactivity
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Fetch API**: HTTP requests to backend

### Machine Learning
- **Classification Algorithm**: Ensemble/tree-based model (saved as pipeline)
- **Feature Engineering**: Custom voltage and current transformations
- **Model Persistence**: Joblib serialization

### AI/LLM Integration
- **RAG System**: Document retrieval for fault solutions
- **Vector Store**: Embeddings-based document storage
- **Builder Agent**: Conversational AI for diagnostic assistance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/KudoroEsther/Machine_Learning_Projects.git
cd Machine_Learning_Projects/Fault_Analysis/Main_fault
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

1. **Start the FastAPI server**
```bash
python main.py
# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. **Open the web interface**
```bash
# Open powerbot.html in your web browser
# Or navigate to http://localhost:8000 if serving via FastAPI
```

3. **Access the API documentation**
```
http://localhost:8000/docs
```

### Testing the API

**Using curl:**
```bash
# Welcome endpoint
curl http://localhost:8000/

# Predict endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Va": 220.5,
    "Vb": 221.0,
    "Vc": 220.8,
    "Ia": 10.2,
    "Ib": 10.5,
    "Ic": 10.3
  }'

# Diagnose endpoint (with AI explanation)
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "Va": 220.5,
    "Vb": 221.0,
    "Vc": 220.8,
    "Ia": 10.2,
    "Ib": 10.5,
    "Ic": 10.3
  }'
```

## 📁 Project Structure

```
Main_fault/
│
├── main.py                          # FastAPI application and endpoints
├── fault_Copy.py                    # Feature engineering class
├── fault_rag_using_utils.py         # RAG system configuration
├── detection_pipeline.pkl           # Trained ML model
├── powerbot.html                    # Web interface
│
├── data/                            # Training/test datasets (if applicable)
├── models/                          # Model training scripts
├── utils/                           # Utility functions
└── docs/                            # Documentation and RAG documents
```

## 🔌 API Reference

### Endpoints

#### 1. Welcome Endpoint
```http
GET /
```
Returns a welcome message.

**Response:**
```json
{
  "message": "Welcome to Transmission Line Fault Predictor"
}
```

#### 2. Predict Endpoint
```http
POST /predict
```
Predicts fault type and confidence score.

**Request Body:**
```json
{
  "Va": float,  // Voltage phase A
  "Vb": float,  // Voltage phase B
  "Vc": float,  // Voltage phase C
  "Ia": float,  // Current phase A
  "Ib": float,  // Current phase B
  "Ic": float   // Current phase C
}
```

**Response (No Fault):**
```json
{
  "status": "no_fault",
  "fault_label": "No fault",
  "confidence": 0.987
}
```

**Response (Fault Detected):**
```json
{
  "status": "fault",
  "fault_label": "LLG fault",
  "confidence": 0.952
}
```

#### 3. Diagnose Endpoint
```http
POST /diagnose
```
Provides AI-powered fault diagnosis with explanations and solutions.

**Request Body:**
```json
{
  "Va": float,
  "Vb": float,
  "Vc": float,
  "Ia": float,
  "Ib": float,
  "Ic": float
}
```

**Response:**
```json
{
  "fault_label": "LLG fault",
  "confidence": 0.952,
  "final_answer": "Detailed AI-generated explanation and recommended solutions..."
}
```

## 🎯 Usage Examples

### Python Client Example
```python
import requests

# Define the endpoint
url = "http://localhost:8000/diagnose"

# Prepare the data
data = {
    "Va": 220.5,
    "Vb": 150.2,  # Abnormal voltage
    "Vc": 220.8,
    "Ia": 10.2,
    "Ib": 25.8,   # High current indicating fault
    "Ic": 10.3
}

# Send request
response = requests.post(url, json=data)
result = response.json()

print(f"Fault Type: {result['fault_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Diagnosis: {result['final_answer']}")
```

### JavaScript/Frontend Example
```javascript
const data = {
    Va: 220.5,
    Vb: 221.0,
    Vc: 220.8,
    Ia: 10.2,
    Ib: 10.5,
    Ic: 10.3
};

fetch('http://localhost:8000/diagnose', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => {
    console.log('Fault:', result.fault_label);
    console.log('Confidence:', result.confidence);
    console.log('Diagnosis:', result.final_answer);
});
```

## 🧪 Model Information

The fault detection model is an ensemble machine learning pipeline that:
- Accepts 6 input features (Va, Vb, Vc, Ia, Ib, Ic)
- Performs feature engineering transformations
- Classifies faults into 4 categories
- Provides probability estimates for predictions

### Input Features
- **Va, Vb, Vc**: Three-phase voltage readings (Volts)
- **Ia, Ib, Ic**: Three-phase current readings (Amperes)

### Output Classes
1. **No fault**: Normal system operation
2. **LG fault**: Single line-to-ground fault
3. **LLG fault**: Double line-to-ground fault
4. **LLLG fault**: Three-phase-to-ground fault

## 📊 Project Status

**Current Version:** 1.0.0

### Completed Features ✅
- [x] Machine learning model training and deployment
- [x] FastAPI backend implementation
- [x] RESTful API endpoints
- [x] RAG-based diagnostic system
- [x] Web-based chatbot interface
- [x] Real-time fault classification
- [x] Confidence scoring

### In Progress 🚧
- [ ] Model performance optimization
- [ ] Extended fault type coverage
- [ ] Historical data logging and analytics
- [ ] Mobile application development

### Future Enhancements 🔮
- [ ] Multi-language support
- [ ] Advanced visualization dashboards
- [ ] Automated alert notifications
- [ ] Integration with SCADA systems
- [ ] Batch processing capabilities
- [ ] Model retraining pipeline
- [ ] Enhanced explainability features (SHAP/LIME)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of the Machine Learning Projects repository. Please refer to the main repository for licensing information.

## 👥 Authors

- **Kudoro Esther** - [GitHub Profile](https://github.com/KudoroEsther)

## 🙏 Acknowledgments

- Power systems domain experts for fault classification guidelines
- Open-source ML community for tools and frameworks
- Contributors to the FastAPI and LangChain ecosystems

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue in the GitHub repository
- Contact the maintainer through GitHub

---

**Note:** This is an educational/research project. For production deployment in critical infrastructure, additional testing, validation, and safety measures are required.