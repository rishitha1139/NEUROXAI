# 🧠 NeuroXAI — Parkinson's Disease Prediction (updated)

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/web-flask-green.svg)
![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205.1.3-purple.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP%2FLIME-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository provides a Flask web application for Parkinson's disease prediction with multiple ML/DL models and Explainable AI (SHAP/LIME) support. The project includes training pipelines, a saved preprocessor (for inference alignment), a responsive Bootstrap UI, and REST endpoints for prediction, explanation, upload, and reporting.

This README has been reconstructed to reflect recent improvements:
- Robust runtime preprocessor alignment (fill missing features, drop unexpected inputs)
- New/updated API endpoints for preprocessor info and repair, and model reload
- Frontend improvements: dynamic model listing and a missing-features modal (Basic / Recommended)
- Deployment notes (Vercel issues with heavy ML packages & recommended workarounds)

---

## Quick Start (Windows)

1. Create and activate a venv

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies (development / full)

```powershell
pip install -r requirements.txt
```

If you want a lightweight deployment without heavy ML libs (TensorFlow / XGBoost), see the "Deployment" section below (use `requirements-lite.txt`).

3. Run the app locally

Run with unbuffered output so logs are streamed in real time (recommended):

```powershell
python -u app.py
```

Open http://127.0.0.1:5000 to use the UI. The server uses lazy imports and background model loading to reduce startup blocking.

---

## What changed (high level)

- Preprocessor alignment
  - The saved preprocessor (`models/preprocessor.pkl`) now stores inference metadata (feature names and feature means) and is used at runtime to:
    - reindex incoming requests to the trained feature order
    - fill missing features with stored means (fallback to 0.0)
    - drop unexpected input fields (logged) to avoid hard failures

- Backend
  - New/updated endpoints:
    - GET /api/preprocessor/info — returns feature names and feature means (safe metadata)
    - POST /api/preprocessor/repair — attempt to refit/repair preprocessor from training CSV
    - POST /api/models/reload — reload model artifacts from `models/` at runtime
  - Lazy imports and a background loader thread are used so the Flask server becomes responsive quickly while heavy model loads happen in the background.

- Frontend
  - `templates/index.html` now dynamically populates the model dropdown from `/api/models`.
  - A "missing-features" modal guides users when the submitted form lacks features required by the preprocessor; it groups defaults into Basic and Recommended values.
  - The UI no longer causes backend errors when extra fields (for example `DoctorInCharge`) are included — the backend will drop them and log a warning.

---

## Project layout (concise)

- `app.py` — Flask app and REST endpoints
- `data/` — datasets (e.g., `parkinsons_disease_data.csv`)
- `models/` — model artifacts and `preprocessor.pkl`
- `src/` — core modules:
  - `preprocessing.py` — DataPreprocessor with `preprocess_pipeline`, `fit_for_inference`, `prepare_inference` and helpers
  - `model_training.py` / `train_and_save_models.py` — training scripts that persist models and preprocessor
  - `explainability.py` — SHAP/LIME wrappers (imported lazily)
  - `generate_results.py` — report generation
- `templates/` — `index.html`, `reports.html`
- `results/` — generated reports (plots, CSV, PDFs)

---

## API (key endpoints and usage)

- GET /api/health
  - Returns basic health, loaded models and preprocessor metadata summary.

- GET /api/models
  - Returns a JSON map of available models found in `models/`.

- POST /api/models/reload
  - Reload models from disk without restarting the server.

- POST /api/preprocessor/repair
  - Attempts to repair or fit the preprocessor from `data/parkinsons_disease_data.csv` and saves it to `models/preprocessor.pkl`.

- GET /api/preprocessor/info
  - Returns safe metadata: `feature_names`, `feature_means`, and basic counts.

- POST /api/predict
  - Body: {"model": "random_forest", "features": {<form-field>: value, ...}}
  - The server maps UI form names to trained feature names, drops unknown keys, fills missing features from `feature_means`, then calls the model's predict/predict_proba.

- POST /api/explain
  - Similar to `/api/predict` but returns SHAP/LIME explanations (if explainer initialized).

- POST /api/upload
  - Upload a CSV for batch prediction (saved to `data/`), then `process_batch_prediction` generates predictions and reports.

Examples (curl)

```cmd
curl -v http://127.0.0.1:5000/api/health

curl -v -X POST http://127.0.0.1:5000/api/models/reload

curl -H "Content-Type: application/json" -d "{\"model\":\"random_forest\",\"features\":{\"Age\":50,\"BMI\":25}}" http://127.0.0.1:5000/api/predict
```

---

## How to train & persist the preprocessor

1. Prepare your dataset at `data/parkinsons_disease_data.csv` with the expected columns used in training.
2. Run the training script (it saves models and `models/preprocessor.pkl`):

```powershell
python src/train_and_save_models.py
```

The training pipeline calls `DataPreprocessor.fit_for_inference()` (or equivalent) so the saved preprocessor contains `feature_names` and `feature_means` used at runtime.

---

## Deployment notes (Vercel & heavy ML packages)

If you deploy to a serverless platform such as Vercel, heavy compiled ML packages often cause build failures (pip compiling C extensions, missing manylinux wheels, TensorFlow binary size, etc.). You may see errors during `pip install -r requirements.txt` for packages like `tensorflow`, `xgboost`, or `scikit-learn` on newer Python versions (e.g., 3.12).

Options to fix deployment issues:

- Use a lightweight requirements file for serverless deploys (`requirements-lite.txt`) that excludes heavy packages and keeps only runtime Flask deps (Flask, joblib, pandas, numpy if available as wheels).
- Pin to Python versions with wheel support (Python 3.11 is commonly safer than 3.12 for some ML wheels). Add `vercel.json` to force `python3.11` for function runtime:

```json
{
  "functions": {
    "api/**/*.py": { "runtime": "python3.11" }
  }
}
```

- Use a Docker-based deployment (Render, Cloud Run, or Docker on Vercel) where you control the build image and system packages.
- If you need XGBoost/TensorFlow in the serverless function, consider hosting only the lightweight web front-end on Vercel and calling an external inference API/endpoint (a VM/container that has ML libs installed).

If your Vercel build fails with pip errors, reproduce locally with a verbose pip install and inspect the log to find the failing package. See Troubleshooting below for commands.

---

## Troubleshooting & debugging

- Reproduce pip install failures locally (capture verbose log):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -v > pip_install_log.txt 2>&1
type pip_install_log.txt
```

- Check server logs for warnings about dropped features (e.g. `DoctorInCharge`) — backend now logs dropped unexpected keys when a request contains fields not seen during training.

- If the server prints TensorFlow logs and you don't use DNNs, remove or move `dnn_model.keras` out of `models/` or guard DNN loading in `app.py` to avoid TF initialization.

---

## Recommended quick fixes (when upgrading / deploying)

- Create `requirements-lite.txt` that contains only the minimal runtime packages required by the Flask app (exclude TF/xgboost). Use that for serverless deployments.
- Pin heavy packages to versions known to provide manylinux wheels for your Python runtime.
- Prefer Docker/Cloud Run when you need native compiled packages in the runtime.

---

## Notes & disclaimer

This project is for research and educational purposes only. Do not use results for clinical decisions without professional validation.

---

## License & author

MIT License — see `LICENSE` file.

Developed by Varun Sallagali © 2025