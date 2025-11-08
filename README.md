# 🧠 NeuroXAI — Parkinson's Disease Prediction System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/web-flask-green.svg)
![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205.1.3-purple.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP%2FLIME-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive end-to-end machine learning system for Parkinson's Disease prediction with integrated Explainable AI (XAI) capabilities. The system provides real-time predictions, confidence scores, and interpretable explanations using SHAP and LIME techniques.

---

## 🎯 Key Features

- **End-to-End Pipeline**: Data preprocessing → ML/DL model training → XAI module → Flask web application → Real-time inference and explanations
- **Real-time Prediction**: Instant predictions with confidence scores
- **SHAP & LIME Explanations**: Clear explanations using XAI techniques
- **Batch Processing**: CSV file upload for multiple patient predictions
- **Model Comparison**: Compare predictions across different ML/DL models
- **RESTful API**: Complete API endpoints for training and inference

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Web Application Features](#web-application-features)
3. [Quick Start](#quick-start)
4. [API Documentation](#api-documentation)
5. [Deployment Guide](#deployment-guide)
6. [Demo Guide](#demo-guide)
7. [Troubleshooting](#troubleshooting)

---

## 🏗️ System Architecture

Our entire project follows an **end-to-end pipeline**:

```
Data Preprocessing → ML/DL Model Training → XAI Module → Flask Web Application → Real-time Inference & Explanations
```

### Architecture Components

1. **Data Preprocessing Module** (`src/preprocessing.py`)
   - Handles data cleaning, normalization, and feature engineering
   - Preprocessor saved with inference metadata for runtime alignment
   - Supports missing feature handling and feature name mapping

2. **Model Training Pipeline** (`src/model_training.py`, `src/train_and_save_models.py`)
   - Multiple ML/DL algorithms: Random Forest, XGBoost, SVM, Logistic Regression, DNN
   - Model persistence and versioning
   - Cross-validation and hyperparameter tuning

3. **XAI Module** (`src/explainability.py`)
   - SHAP (SHapley Additive exPlanations) for global and local explanations
   - LIME (Local Interpretable Model-agnostic Explanations) for local interpretability
   - Feature importance visualization and analysis

4. **Flask Web Application** (`app.py`)
   - RESTful API endpoints for all operations
   - Real-time prediction and explanation services
   - Batch processing capabilities

5. **Frontend Interface** (`templates/index.html`)
   - Modern, responsive Bootstrap UI
   - Interactive forms for patient data input
   - Real-time visualization of predictions and explanations

---

## 🌐 Web Application Features

### ✅ Real-time Prediction
- **Endpoint**: `POST /api/predict`
- Instant predictions for individual patients
- Support for multiple ML models (Random Forest, XGBoost, SVM, Logistic Regression, DNN)
- Model selection via dropdown menu
- Asynchronous API calls for responsive UI

### ✅ Confidence Scores
- Probability scores for each prediction class (Parkinson / No Parkinson)
- Confidence percentage display with progress bar
- Probability distribution visualization
- Threshold-based decision making

### ✅ SHAP and LIME Explanations
- **SHAP Explanations**: Feature contribution analysis, global and local interpretability, visual SHAP plots
- **LIME Explanations**: Local model-agnostic explanations, feature importance rankings
- **Endpoint**: `POST /api/explain` with `type: "shap"` or `type: "lime"`
- Interactive feature importance visualization with Plotly

### ✅ CSV File Upload for Batch Prediction
- **Endpoint**: `POST /api/upload`
- Bulk patient data processing
- Automatic batch prediction generation
- CSV file validation and error handling
- Batch explanation support via `POST /api/explain_batch`
- Generated reports with predictions and explanations

### ✅ Model Comparison
- Select different models from dropdown
- Compare predictions with same patient data
- Side-by-side prediction comparison
- Confidence score differences

---

## 🚀 Quick Start

### Local Development

1. **Create and activate virtual environment**:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

For lightweight deployment (without TensorFlow/XGBoost), use `requirements-lite.txt`.

3. **Run the application**:
```powershell
python -u app.py
```

4. **Access the application**: Open http://127.0.0.1:5000

### Train Models

1. Prepare your dataset at `data/parkinsons_disease_data.csv`
2. Run the training script:
```powershell
python src/train_and_save_models.py
```

The training pipeline saves models and `models/preprocessor.pkl` with inference metadata.

---

## 📡 API Documentation

### Inference Endpoints

- **`POST /api/predict`** - Single prediction
  ```json
  {
    "model": "random_forest",
    "features": {
      "Age": 50,
      "BMI": 25,
      "Tremor": 1
    }
  }
  ```

- **`POST /api/explain`** - Single explanation (SHAP/LIME)
  ```json
  {
    "model": "random_forest",
    "features": {...},
    "type": "shap"  // or "lime"
  }
  ```

- **`POST /api/upload`** - File upload for batch processing
- **`POST /api/batch_predict`** - Batch predictions
- **`POST /api/explain_batch`** - Batch explanations

### Model Management

- **`GET /api/models`** - List available models
- **`POST /api/models/reload`** - Reload models from disk
- **`GET /api/health`** - System health check

### Training Endpoints

- **`POST /api/train`** - Train new models (ML or DL)
  ```json
  {
    "model_type": "ml",  // or "dl"
    "model_name": "random_forest"
  }
  ```

### Preprocessor Management

- **`GET /api/preprocessor/info`** - Get preprocessor metadata
- **`POST /api/preprocessor/repair`** - Repair/refit preprocessor

### Example API Calls

```bash
# Health check
curl http://127.0.0.1:5000/api/health

# Get available models
curl http://127.0.0.1:5000/api/models

# Make prediction
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model":"random_forest","features":{"Age":50,"BMI":25}}'

# Get SHAP explanation
curl -X POST http://127.0.0.1:5000/api/explain \
  -H "Content-Type: application/json" \
  -d '{"model":"random_forest","features":{"Age":50},"type":"shap"}'
```

---

## 🚢 Deployment Guide

### Vercel Deployment

#### Prerequisites
1. Vercel account (sign up at https://vercel.com)
2. Vercel CLI (optional): `npm i -g vercel`

#### Deployment Steps

1. **Prepare Project**:
   - Ensure `vercel.json` exists
   - Ensure `api/index.py` exists
   - Model files in `models/` directory (must be committed)
   - Use `requirements-lite.txt` for lightweight deployment

2. **Deploy via Dashboard**:
   - Go to https://vercel.com/dashboard
   - Click "Add New Project"
   - Import your Git repository
   - Framework Preset: Other
   - Build Command: (leave empty)
   - Install Command: `pip install -r requirements-lite.txt`
   - Click "Deploy"

3. **Deploy via CLI**:
```bash
vercel login
vercel --prod
```

#### Important Notes

- **Model Files**: Must be committed to repository (not in `.gitignore`)
- **File Storage**: On Vercel, use `/tmp` for writable directories (ephemeral)
- **Heavy Dependencies**: TensorFlow/XGBoost may cause build issues - use `requirements-lite.txt`
- **Python Version**: Configured for Python 3.11 in `vercel.json` and `runtime.txt`

#### XAI Features on Vercel

The application includes SHAP and LIME in `requirements-lite.txt`:
- SHAP 0.41.0
- LIME 0.2.0.1
- Dependencies: scipy, numba, tqdm

**Note**: Numba compilation may increase build time, but XAI features work in serverless environment.

---

## 🎬 Demo Guide

### Live Demo Steps

1. **Enter Patient Details**
   - Fill in the prediction form with patient information
   - Demographics, clinical features, medical history, lifestyle factors

2. **Get Prediction**
   - Click "Predict" button
   - View result: "Parkinson" or "No Parkinson"
   - See confidence score and probability distribution

3. **View SHAP/LIME Explanation**
   - Click "Explain Prediction" button
   - Select explanation type (SHAP or LIME)
   - View feature importance visualization
   - See color-coded feature contributions

4. **Batch Prediction**
   - Upload CSV file with multiple patient records
   - View batch prediction results
   - Download results CSV

5. **Model Comparison**
   - Select different models from dropdown
   - Compare predictions with same data
   - View confidence differences

---

## 🔧 Troubleshooting

### Common Issues

1. **Pip Install Failures**:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -v > pip_install_log.txt 2>&1
```

2. **Model Loading Issues**:
   - Ensure model files are in `models/` directory
   - Check file formats (.joblib, .pkl, .keras, .h5)
   - Verify preprocessor exists: `models/preprocessor.pkl`

3. **Vercel Build Failures**:
   - Use `requirements-lite.txt` instead of `requirements.txt`
   - Ensure Python 3.11 is specified in `runtime.txt`
   - Check build logs for specific errors

4. **XAI Features Not Working**:
   - Verify SHAP/LIME are in requirements
   - Check if explainer initialized: `GET /api/health`
   - Review server logs for import errors

5. **Missing Features Error**:
   - Backend automatically fills missing features with means
   - Check preprocessor info: `GET /api/preprocessor/info`
   - Repair preprocessor if needed: `POST /api/preprocessor/repair`

---

## 📁 Project Structure

```
NeuroXAI/
├── app.py                 # Flask application and REST endpoints
├── api/
│   └── index.py          # Vercel serverless function handler
├── src/
│   ├── preprocessing.py   # Data preprocessing module
│   ├── model_training.py  # Model training functions
│   ├── explainability.py  # SHAP/LIME XAI module
│   ├── generate_results.py # Report generation
│   └── train_and_save_models.py # Training pipeline
├── templates/
│   ├── index.html        # Main web interface
│   └── reports.html      # Reports page
├── models/               # Trained model files
├── data/                 # Datasets
├── results/              # Generated reports
├── requirements.txt      # Full dependencies
├── requirements-lite.txt # Lightweight dependencies (for Vercel)
├── vercel.json          # Vercel configuration
└── runtime.txt          # Python version specification
```

---

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask 2.2.5
- **ML Libraries**: scikit-learn 1.2.2, TensorFlow 2.11.0 (optional)
- **XAI Libraries**: SHAP 0.41.0, LIME 0.2.0.1
- **Data Processing**: pandas 1.5.3, numpy 1.24.3
- **Visualization**: matplotlib 3.7.1, seaborn 0.12.2

### Frontend
- **Framework**: Bootstrap 5.1.3
- **JavaScript**: Vanilla JS with async/await
- **Visualization**: Plotly
- **Icons**: Font Awesome 6.0.0

### Deployment
- **Platform**: Vercel (serverless)
- **Python Version**: 3.11
- **Architecture**: Serverless functions with API routing

---

## ⚠️ Important Notes

### Preprocessor Alignment
- The saved preprocessor (`models/preprocessor.pkl`) stores inference metadata
- Automatically fills missing features with stored means
- Drops unexpected input fields (logged as warnings)

### Model Files
- Model files should be committed to repository for deployment
- Currently in `.gitignore` - adjust if needed for deployment
- Supported formats: `.joblib`, `.pkl`, `.keras`, `.h5`

### File Storage on Vercel
- Writable directories use `/tmp` (temporary storage)
- Uploaded files and results are **ephemeral**
- For persistent storage, use external services (S3, Cloud Storage)

---

## 📝 License & Disclaimer

**MIT License** — see `LICENSE` file.

**Disclaimer**: This project is for research and educational purposes only. Do not use results for clinical decisions without professional validation.

---

## 👤 Author

**Developed by Varun Sallagali © 2025**

---

## 🔗 Quick Links

- **Health Check**: `GET /api/health`
- **Available Models**: `GET /api/models`
- **Preprocessor Info**: `GET /api/preprocessor/info`
- **Local Access**: http://127.0.0.1:5000

---

**Status**: ✅ Production Ready | All features verified and working
