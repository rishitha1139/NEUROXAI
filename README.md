# 🧠 NeuroXAI - Parkinson's Disease Prediction using Explainable AI

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/web-flask-green.svg)
![TensorFlow](https://img.shields.io/badge/deep--learning-tensorflow-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive machine learning and deep learning framework for Parkinson's disease diagnosis with **state-of-the-art Explainable AI (XAI)** techniques.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurodegenerative disorder that primarily affects movement, leading to symptoms like tremors, stiffness, and difficulty with balance and coordination.  

This project applies **Explainable AI** to make predictions transparent, interpretable, and clinically trustworthy.  

---

## 🏗️ Project Structure

NeuroXAI/
│── data/ # Dataset storage
│ └── parkinsons_disease_data.csv
│── notebooks/ # Jupyter notebooks for exploration
│── src/ # Source code
│ ├── preprocessing.py # Data cleaning and preprocessing
│ ├── feature_selection.py # Feature importance and selection
│ ├── model_training.py # ML/DL model training
│ ├── explainability.py # SHAP, LIME, and XAI techniques
│ └── utils.py # Visualization and utilities
│── models/ # Trained models (created after training)
│── results/ # Model results and reports
│── app.py # Flask web application
│── requirements.txt # Python dependencies
│── README.md # This file


---

## 🚀 Features

### 🔹 Core ML/DL Capabilities
- Traditional ML: **Random Forest, SVM, Logistic Regression, Gradient Boosting, XGBoost**
- Deep Learning: **TensorFlow/Keras-based DNN**
- Feature Engineering: Feature importance + selection
- Model Evaluation: Accuracy, Precision, Recall, F1, ROC

### 🔹 Explainable AI (XAI)
- **SHAP**: Global + local interpretability
- **LIME**: Local prediction explanations
- **Feature Importance**: Multiple methods
- **Confidence Scores**: Model prediction reliability

### 🔹 Web Application
- RESTful API for predictions
- File upload for **batch predictions**
- Real-time SHAP & LIME explanations
- Model training & evaluation via API

---

## 🎥 Demo

👉 Run the app:

- python app.py
- Then open http://localhost:5000/ in your browser.

---

##📋 Requirements

Python: 3.8 or higher

Install dependencies:

pip install -r requirements.txt

# Clone repo
git clone https://github.com/VarunSallagali/NeuroXAI.git
cd NeuroXAI

# Create venv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run app
python app.py

## 📊 Dataset Information

Samples: ~2100 patients

Features: Clinical, lifestyle, and neurological features

Target: Binary classification (0 = No Parkinson, 1 = Parkinson)

Includes: Age, BMI, Lifestyle habits, Clinical measures, Neurological assessments

⚠️ Note: Our dataset is an extended version of the UCI Parkinson’s dataset, enriched with clinical and lifestyle features.

## 🌐 Web Application API
Endpoints

GET /api/health → Check app status
GET /api/models → List available models
POST /api/predict → Single prediction
POST /api/batch_predict → Batch predictions
POST /api/explain → Explain single prediction
POST /api/explain_batch → Explain batch predictions
POST /api/upload → Upload CSV data
POST /api/train → Train new models

## 📈 Model Performance

Accuracy: 95%+
Precision: 94%+
Recall: 96%+
F1-Score: 95%+

## 🔬 Research Applications

Clinical decision support

Patient-friendly diagnosis explanations

Parkinson’s biomarker research

Trustworthy AI in healthcare

## 📚 References

SHAP Documentation

LIME Documentation

Parkinson's Dataset - UCI

Explainable AI in Healthcare

## 📄 License

This project is licensed under the MIT License – see the LICENSE
 file.


 ## 👨‍💻 Author

Developed by Varun Sallagali

📌 Capstone Project | Placement Preparation | AI + XAI in Healthcare

⚠️ Disclaimer: This is a research tool. It should not be used for clinical diagnosis without proper medical validation.

