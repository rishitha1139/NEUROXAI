
# 🧠 NeuroXAI — Parkinson's Disease Prediction 

Lightweight, reproducible Flask app for Parkinson's disease prediction with XAI (SHAP/LIME) support.

This repository contains a complete pipeline: data preprocessing → model training → explainability → web app → result generation.

Quick summary
- App: `app.py` — Flask server with REST endpoints for prediction, explanation, model management and uploads.
- Training: `src/train_and_save_models.py` — trains models and writes artifacts to `models/`.
- Preprocessing: `src/preprocessing.py` — centralized preprocessing and inference alignment (saves `preprocessor.pkl`).
- Results: `src/generate_results.py` — regenerates evaluation plots using the saved test set and preprocessor.

If you previously had a separate QUICKSTART.md, this README now contains the single, canonical Quick Start and run instructions.

---

## Quick Start (Local development)

These steps assume you're on Windows (PowerShell or cmd) and have a Python 3.8–3.11 virtual environment.

1) Create & activate venv

PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows cmd
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

2) Install dependencies

Use the full dependencies for development (may include heavy packages like TensorFlow and XGBoost):
```powershell
pip install -r requirements.txt
```

For lightweight serverless deploys (no TensorFlow/XGBoost), use:
```powershell
pip install -r requirements-lite.txt
```

3) Run the Flask app (development)

Preferred (avoids import-as-script issues):
```powershell
python -m app
```
or directly:
```powershell
python app.py
```

Visit: http://127.0.0.1:5000

4) Train models (create artifacts in `models/`)

Run the training pipeline (recommended using module mode so imports resolve):
```powershell
python -m src.train_and_save_models
```

This will:
- preprocess the dataset at `data/parkinsons_disease_data.csv`
- save `models/preprocessor.pkl` with feature metadata
- save model artifacts (`models/*.joblib`, `models/dnn_model.keras`)
- persist test indices to `models/test_indices.joblib` for reproducible evaluation

5) Reproduce evaluation plots

After training, regenerate plots using the same test set and preprocessor:
```powershell
python -m src.generate_results
```
Saved plots appear under `results/`.

---

## Making runs reproducible (recommended)

- Set deterministic seeds at script start (done in the training script). For full reproducibility across systems set these env vars before running:
```cmd
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONHASHSEED=0
```
- Use `python -m` to run package modules (avoids ModuleNotFoundError).
- The training script saves `models/test_indices.joblib` and `models/preprocessor.pkl` — `generate_results.py` will reuse these to ensure plots match training runs.

---

## API (short)

- `GET /api/health` — health check
- `GET /api/models` — list available models
- `POST /api/predict` — single prediction (JSON: `{ "model":"random_forest", "features": {...} }`)
- `POST /api/explain` — get SHAP/LIME explanation (include `type`)
- `POST /api/upload` — batch CSV upload
- `POST /api/models/reload` — reload models from disk
- `GET /api/preprocessor/info` — preprocessor metadata

---

## Troubleshooting (common)

- Pip install fails: upgrade wheel/setuptools then retry:
```cmd
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
- ModuleNotFoundError when running training: run as module from repo root:
```cmd
python -m src.train_and_save_models
```
- If models fail to load (XGBoost/TensorFlow missing): install the corresponding package or use `requirements-lite.txt` and avoid DL models.
- If plots change between runs: ensure you re-run training (which saves the test indices) and re-run `src.generate_results.py`; also set the recommended env variables above.

---

## Project structure (key files)

```
NeuroXAI/
├── app.py
├── src/
│   ├── preprocessing.py
│   ├── train_and_save_models.py
│   ├── generate_results.py
│   └── explainability.py
├── templates/
├── models/
├── data/
├── results/
├── requirements.txt
├── requirements-lite.txt
└── README.md
```

---

If you'd like, I can also:
- remove mentions of QUICKSTART.md from the repo (none found) and ensure the README is the single authoritative run guide,
- add a short `CONTRIBUTING.md` or `RUNNING.md` with exact commands for CI/deployment,
- or open a PR with these changes.

If this looks good, I will mark the README update completed and can (optionally) commit a small CONTRIBUTING note and a short test script to validate installs.

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

**Developed by Challa Sai Rishitha © 2025**

---

## 🔗 Quick Links

- **Health Check**: `GET /api/health`
- **Available Models**: `GET /api/models`
- **Preprocessor Info**: `GET /api/preprocessor/info`
- **Local Access**: http://127.0.0.1:5000

---

**Status**: ✅ Production Ready | All features verified and working
