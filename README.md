# 🧠 NeuroXAI - Parkinson's Disease Prediction using Explainable AI

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/web-flask-green.svg)
![TensorFlow](https://img.shields.io/badge/deep--learning-tensorflow-orange.svg)
![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205.1.3-purple.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP%2FLIME-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A modern web application for Parkinson's disease prediction using machine learning and deep learning models, enhanced with Explainable AI (SHAP, LIME). Features an intuitive Bootstrap UI, comprehensive visualizations, and detailed model explanations through an interactive web interface.

---

## Key additions in this README
- How to ensure model/preprocessor feature consistency
- How to regenerate models and reports
- How to view generated reports from the web app (/reports)

---

## Quick Start (Windows)

1. Create and activate venv
```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies
```powershell
pip install -r requirements.txt
```

3. Train models and save the fitted preprocessor
```powershell
python src/train_and_save_models.py
```
This fits and saves the preprocessor (scaler, imputer, feature names) and trained models in `models/`.

4. Generate results / reports (plots, CSV, PDF) into `results/`
```powershell
python src/generate_results.py
```

5. Run the Flask app and view reports
```powershell
python app.py
```
Open http://127.0.0.1:5000/reports to list and preview generated report files.

---

## Project Structure

NeuroXAI/
- data/ — datasets (parkinsons_disease_data.csv)
- src/ — preprocessing, training, explainability, utilities
  - explainability.py — SHAP and LIME implementations
  - feature_selection.py — Feature importance analysis
  - model_training.py — Model training pipelines
  - preprocessing.py — Data preprocessing and transformation
  - generate_results.py — Report generation utilities
  - utils.py — Helper functions and utilities
- models/ — saved models and preprocessor (.pkl, .h5)
  - preprocessor.pkl — Fitted preprocessor with feature names
  - dnn_model.keras — Deep Neural Network model
  - random_forest_model.joblib — Random Forest model
  - xgboost_model.joblib — XGBoost model
  - svm_model.joblib — SVM model
  - logistic_model.joblib — Logistic Regression model
- results/ — generated plots and report files
  - confusion_matrices.png — Model accuracy visualization
  - model_comparison.png — Performance comparison plots
  - roc_curves.png — ROC curves for all models
  - shap_importance.png — SHAP feature importance plots
  - xgb_feature_importance.png — XGBoost feature importance
  - rf_feature_importance.png — Random Forest feature importance
- templates/ — Flask HTML templates with modern UI
  - index.html — Main prediction interface
  - reports.html — Analysis reports dashboard
- app.py — Flask application with REST endpoints

---

## Reports in the Web App

- Ensure `src/generate_results.py` writes outputs to project/results/.
- The Flask app should expose:
  - GET /reports → HTML page listing files in `results/`
  - GET /results/<filename> → Serve specific report file

If images or PDFs do not appear, check:
- `results/` exists and contains files
- File permissions and Flask logs for 404 errors
- URL paths in browser (use the exact filename)

---

## Preprocessor & Feature Consistency (Important)

- Always retrain and resave the preprocessor when feature set changes:
  1. Run training script: `python src/train_and_save_models.py`
  2. This updates `models/preprocessor.pkl` (contains scaler, imputer, feature_names).
  3. Any prediction or report generation must load this same preprocessor.
- For runtime predictions, use a transform method that:
  - applies the fitted imputer and scaler
  - reindexes incoming data to the saved feature order (adds missing cols with default values)

If you see: `The feature names should match those that were passed during fit.` — retrain and resave the preprocessor and models.

---

## API Endpoints (summary)

- GET /api/health → Health check  
- GET /api/models → List available models  
- POST /api/predict → Single prediction  
- POST /api/batch_predict → Batch predictions (CSV)  
- POST /api/explain → Explain single prediction (SHAP/LIME)  
- POST /api/explain_batch → Explain batch predictions  
- POST /api/upload → Upload CSV  
- POST /api/train → Trigger retrain (if implemented)  
- GET /reports → List and preview files in `results/`  
- GET /results/<filename> → Serve report file

---

## Troubleshooting

### Model Issues
- Feature name errors → Retrain and save preprocessor; ensure CSV columns match training data
- DNN shape errors → Check input reshaping matches training shape (e.g., (n_samples, n_features, 1))
- Prediction inconsistency → Verify preprocessor.pkl is latest version

### Web Interface
- Reports not loading → Check `results/` folder permissions and Flask file serving
- Visualizations broken → Ensure all plot files exist in `results/` directory
- UI elements misaligned → Clear browser cache or check console for Bootstrap/JS errors

### Development
- Training errors → Verify dataset format and feature engineering steps
- Report generation fails → Check write permissions and file paths
- API errors → Monitor Flask logs and verify endpoint parameters

## Notes & Disclaimer

This project is intended for research and educational purposes only. The predictions and analyses should not be used for clinical diagnosis without proper medical validation and supervision.

Features:
- Modern web interface with Bootstrap 5.1.3
- Comprehensive ML/DL model suite
- Extensive visualization capabilities
- Detailed XAI implementations
- RESTful API architecture

## License & Author

MIT License — see LICENSE file for details.

Developed with ❤️ by Varun Sallagali
© 2025 NeuroXAI Project