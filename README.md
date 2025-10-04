# 🧠 NeuroXAI - Parkinson's Disease Prediction using Explainable AI

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/web-flask-green.svg)
![TensorFlow](https://img.shields.io/badge/deep--learning-tensorflow-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning + deep learning framework for Parkinson's disease prediction with Explainable AI (SHAP, LIME) and a Flask web UI that serves predictions and generated reports.

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

## Project layout

NeuroXAI/
- data/ — datasets (parkinsons_disease_data.csv)
- src/ — preprocessing, training, explainability, utilities
- models/ — saved models and preprocessor (.pkl, .h5)
- results/ — generated plots and report files (images, pdf, csv)
- templates/ — Flask HTML templates (add `reports.html` if not present)
- app.py — Flask app (now includes /reports and /results/<filename> endpoints)

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

- Prediction errors about feature names → retrain and save preprocessor; ensure incoming CSV columns match or are aligned by the preprocessor transform.
- Reports not visible → verify `results/` contains files and Flask serves that folder (absolute path recommended).
- DNN shape errors → for DNN models, ensure input is reshaped as during training (e.g., (n_samples, n_features, 1)) before predict().

---

## Notes & Disclaimer

This project is for research/educational use. Not for clinical diagnosis without validation.

---

## License & Author

MIT License — see LICENSE

Developed by Varun Sallagali