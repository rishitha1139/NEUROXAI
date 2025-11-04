"""
Flask web application for Parkinson's Disease Prediction using XAI.
Provides a web interface for model inference and explanations.
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.preprocessing import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.model_training import MLModelTrainer, DLModelTrainer
from src.explainability import ModelExplainer
from src.utils import DataVisualizer, ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for loaded models and preprocessors
models = {}
preprocessor = None
feature_selector = None
explainer = None

# Configuration
MODELS_DIR = 'models'
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_models():
    """Load trained models from disk."""
    global models, preprocessor, feature_selector
    
    try:
        # Load or initialize preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        data_file = os.path.join(DATA_DIR, 'parkinsons_disease_data.csv')
        
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded successfully from %s", preprocessor_path)
        else:
            logger.warning("No preprocessor found at %s. Creating and fitting new preprocessor...", preprocessor_path)
            if os.path.exists(data_file):
                # Initialize and fit preprocessor with training data
                preprocessor = DataPreprocessor()
                data = pd.read_csv(data_file)
                # Get features (all columns except 'status' if it exists)
                features = data.drop(columns=['status']) if 'status' in data.columns else data
                # Fit the preprocessor with the features
                preprocessor.scale_features(features)
                # Save the fitted preprocessor
                joblib.dump(preprocessor, preprocessor_path)
                logger.info("New preprocessor fitted and saved to %s", preprocessor_path)
            else:
                logger.error(f"Training data not found at {data_file}")
                raise FileNotFoundError(f"Training data not found at {data_file}")
        
        # Load feature selector
        feature_selector_path = os.path.join(MODELS_DIR, 'feature_selector.pkl')
        if os.path.exists(feature_selector_path):
            feature_selector = joblib.load(feature_selector_path)
            logger.info("Feature selector loaded successfully")

        # Load ML models (support both .joblib and legacy .pkl extensions)
        ml_models = ['random_forest', 'xgboost', 'svm', 'logistic']
        for model_name in ml_models:
            # Prefer .joblib, fall back to .pkl
            for ext in ('.joblib', '.pkl'):
                model_path = os.path.join(MODELS_DIR, f'{model_name}_model{ext}')
                if os.path.exists(model_path):
                    try:
                        models[model_name] = joblib.load(model_path)
                        logger.info(f"Model {model_name} loaded successfully from {model_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {model_path}: {e}")
        
        # Load DNN model if available
        # Load DNN model if available (support .keras and .h5 extensions)
        for dnn_file in ('dnn_model.keras', 'dnn_model.h5', 'dnn_model.kerasmodel'):
            dnn_path = os.path.join(MODELS_DIR, dnn_file)
            if os.path.exists(dnn_path):
                try:
                    from tensorflow import keras
                    models['dnn'] = keras.models.load_model(dnn_path)
                    logger.info(f"DNN model loaded successfully from {dnn_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load DNN model {dnn_path}: {e}")
        
        logger.info(f"Loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")


def ensure_preprocessor_fitted():
    """Ensure a fitted preprocessor is available globally; try to load or fit from training CSV."""
    global preprocessor
    try:
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        data_file = os.path.join(DATA_DIR, 'parkinsons_disease_data.csv')

        if preprocessor and getattr(preprocessor, 'scaler', None) is not None and getattr(preprocessor, 'feature_names', None):
            return True

        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)  
            logger.info("Preprocessor reloaded from disk")
            return True

        if os.path.exists(data_file):
            logger.info("Fitting new preprocessor from training data")
            p = DataPreprocessor()
            # This will fit scaler and set feature names
            try:
                p.preprocess_pipeline(data_file)
            except Exception:
                # preprocess_pipeline may attempt to split etc; fallback to scale_features directly
                data = p.load_data(data_file)
                p.handle_missing_values(data)
                p.scale_features(data)
            joblib.dump(p, preprocessor_path)
            preprocessor = p
            logger.info("New preprocessor fitted and saved")
            return True

    except Exception as e:
        logger.warning(f"Could not ensure preprocessor fitted: {e}")
    return False

def initialize_explainer():
    """Initialize the model explainer with the best model."""
    global explainer
    
    if models:
        # Use the first available model for explanations
        best_model = list(models.values())[0]
        explainer = ModelExplainer(best_model)
        logger.info("Model explainer initialized")

@app.route('/')
def index():
    """Main page of the application."""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """Get list of available models."""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            'type': type(model).__name__,
            'name': name
        }
    return jsonify(model_info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions using the selected model."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'random_forest')
        features = data.get('features', {})
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        # Map form field names to expected feature names
        feature_mapping = {
            'Age': 'Age',
            'BMI': 'BMI',
            'PhysicalActivity': 'Physical Activity',
            'SystolicBP': 'Blood Pressure (Systolic)',
            'DiastolicBP': 'Blood Pressure (Diastolic)',
            'SleepQuality': 'Sleep Quality',
            'Tremor': 'Tremor',
            'Rigidity': 'Rigidity',
            'Bradykinesia': 'Bradykinesia',
            'PosturalInstability': 'Postural Instability',
            'FamilyHistoryParkinsons': 'Family History',
            'Depression': 'Depression',
            'Hypertension': 'Hypertension'
        }
        
        # Create a new dictionary with mapped feature names
        mapped_features = {}
        for form_name, value in features.items():
            if form_name in feature_mapping:
                mapped_features[feature_mapping[form_name]] = 1 if isinstance(value, bool) and value else (0 if isinstance(value, bool) else float(value))
        
        logger.info(f"Mapped features: {mapped_features}")
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([mapped_features])

        # Transform features for prediction using fitted preprocessor (use prepare_inference if available)
        if preprocessor:
            try:
                # Ensure preprocessor is fitted/loaded
                if not ensure_preprocessor_fitted():
                    raise RuntimeError('Preprocessor not fitted and could not be created')

                # Preferred method: prepare_inference
                if hasattr(preprocessor, 'prepare_inference'):
                    feature_df = preprocessor.prepare_inference(feature_df)
                else:
                    # Fallback for older preprocessor implementations
                    feature_df = preprocessor.transform_for_prediction(feature_df)

                logger.info("Features prepared for prediction successfully")
            except Exception as preprocess_error:
                # Fallback logging and attempt best-effort transform
                logger.warning(f"Preprocessor error during prepare_inference: {preprocess_error}")
                try:
                    feature_df = preprocessor.transform_for_prediction(feature_df)
                    logger.info("Fallback transform_for_prediction succeeded")
                except Exception as e2:
                    logger.error(f"Error preprocessing features: {e2}")
                    return jsonify({'error': 'Error preprocessing features', 'details': str(e2)}), 500
            
        # Make prediction
        try:
            model = models[model_name]
            logger.info(f"Using model: {model_name}")
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(feature_df)[0]
                prediction = model.predict(feature_df)[0]
            else:
                prediction = model.predict(feature_df)[0]
                prediction_proba = [1.0, 0.0] if prediction == 1 else [0.0, 1.0]
        except Exception as model_error:
            logger.error(f"Error making prediction: {model_error}")
            return jsonify({'error': 'Error making prediction'}), 500
        
        result = {
            'prediction': int(prediction),
            'prediction_class': 'Parkinson' if prediction == 1 else 'No Parkinson',
            'confidence': max(prediction_proba),
            'probabilities': {
                'No Parkinson': float(prediction_proba[0]),
                'Parkinson': float(prediction_proba[1])
            },
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

# Add to app.py
@app.route('/reports')
def reports():
    """List and display generated reports."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    files = []
    for f in os.listdir(results_dir):
        if os.path.isfile(os.path.join(results_dir, f)):
            files.append(f)
    return render_template('reports.html', files=files)

@app.route('/results/<path:filename>')
def results_file(filename):
    """Serve files from results directory."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    return send_from_directory(results_dir, filename)
@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Generate explanations for a prediction."""
    try:
        data = request.get_json()
        model_name = data.get('model', 'random_forest')
        features = data.get('features', {})
        explanation_type = data.get('type', 'shap')
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        if not explainer:
            return jsonify({'error': 'Model explainer not initialized'}), 400
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])

        # Preprocess features if preprocessor is available
        if preprocessor:
            try:
                if not ensure_preprocessor_fitted():
                    raise RuntimeError('Preprocessor not fitted and could not be created')

                if hasattr(preprocessor, 'prepare_inference'):
                    feature_df = preprocessor.prepare_inference(feature_df)
                else:
                    feature_df = preprocessor.transform_for_prediction(feature_df)

            except Exception as preprocess_error:
                logger.warning(f"Preprocessor error during explanation prepare: {preprocess_error}")
                try:
                    feature_df = preprocessor.transform_for_prediction(feature_df)
                except Exception as e2:
                    logger.error(f"Error preprocessing features for explanation: {e2}")
                    return jsonify({'error': 'Error preprocessing features for explanation', 'details': str(e2)}), 500

        # Ensure explainer has feature names if possible
        try:
            if hasattr(preprocessor, 'feature_names') and preprocessor.feature_names:
                explainer.set_feature_names(preprocessor.feature_names)
        except Exception:
            # non-fatal
            pass
        
        # Generate explanations
        if explanation_type == 'shap':
            explanation = explainer.explain_with_shap(feature_df, sample_idx=0)
        elif explanation_type == 'lime':
            explanation = explainer.explain_with_lime(feature_df, sample_idx=0)
        else:
            return jsonify({'error': f'Explanation type {explanation_type} not supported'}), 400
        
        if explanation:
            # Convert numpy arrays to lists for JSON serialization
            try:
                if 'shap_values' in explanation and hasattr(explanation['shap_values'], 'tolist'):
                    explanation['shap_values'] = explanation['shap_values'].tolist()
            except Exception:
                pass
            try:
                if 'sample' in explanation and hasattr(explanation['sample'], 'to_dict'):
                    explanation['sample'] = explanation['sample'].to_dict('records')
            except Exception:
                pass

            return jsonify({
                'explanation': explanation,
                'type': explanation_type,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Could not generate explanation'}), 500
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for batch predictions."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        file.save(filepath)
        
        # Load and preprocess data
        data = pd.read_csv(filepath)
        
        # Return data summary
        summary = {
            'filename': filename,
            'rows': len(data),
            'columns': len(data.columns),
            'features': list(data.columns),
            'message': 'File uploaded successfully'
        }
        
        # Also trigger batch prediction automatically
        try:
            batch_result = process_batch_prediction(filename, 'random_forest', data)
            summary['prediction_results'] = batch_result
        except Exception as e:
            logger.warning(f"Automatic batch prediction failed: {e}")
            summary['prediction_warning'] = str(e)
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({'error': str(e)}), 500

def process_batch_prediction(filename, model_name, data=None):
    """Process batch predictions and generate reports."""
    if model_name not in models:
        raise ValueError(f'Model {model_name} not available')
    
    # Load the data if not provided
    if data is None:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError('File not found')
        data = pd.read_csv(filepath)
    
    # Handle different column naming conventions
    expected_columns = ['Age', 'BMI', 'Physical Activity', 'Blood Pressure (Systolic)',
                       'Blood Pressure (Diastolic)', 'Sleep Quality', 'Tremor', 'Rigidity',
                       'Bradykinesia', 'Postural Instability']
    
    # Create feature matrix
    X = pd.DataFrame()
    for col in expected_columns:
        if col in data.columns:
            X[col] = data[col]
        else:
            # Try alternative names
            alt_name = col.lower().replace(' ', '_')
            if alt_name in data.columns:
                X[col] = data[alt_name]
            else:
                logger.warning(f"Column {col} not found in data")
                X[col] = 0  # Use default value
    
    # Preprocess features
    if preprocessor:
        try:
            if hasattr(preprocessor, 'prepare_inference'):
                X = preprocessor.prepare_inference(X)
            else:
                X = preprocessor.transform_for_prediction(X)
        except Exception as e:
            logger.warning(f"Preprocessor failed for batch input, attempting fallback: {e}")
            try:
                X = preprocessor.transform_for_prediction(X)
            except Exception as e2:
                logger.error(f"Batch preprocessing failed: {e2}")
                raise
    
    # Make predictions
    model = models[model_name]
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    
    # Generate reports
    from src.generate_results import generate_prediction_report
    report_paths = generate_prediction_report(model, X, y_pred, y_pred_proba, RESULTS_DIR)
    
    return {
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
        'report_files': report_paths
    }

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions on uploaded data."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_name = data.get('model', 'random_forest')
        
        result = process_batch_prediction(filename, model_name)
        return jsonify({'message': 'Batch prediction completed', 'result': result})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain_batch', methods=['POST'])
def explain_batch():
    """Generate explanations for batch predictions."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_name = data.get('model', 'random_forest')
        sample_indices = data.get('sample_indices', [0, 1, 2])  # Explain first 3 samples
        
        if model_name not in models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        if not explainer:
            return jsonify({'error': 'Model explainer not initialized'}), 400
        
        # Load the uploaded file
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 400
        
        data = pd.read_csv(filepath)
        X = data.drop(columns=['status'])
        
        # Preprocess features if preprocessor is available
        if preprocessor:
            try:
                X_raw = data.drop(columns=['status']) if 'status' in data.columns else data
                if hasattr(preprocessor, 'prepare_inference'):
                    X = preprocessor.prepare_inference(X_raw)
                else:
                    X = preprocessor.scale_features(X_raw, target_column=None, fit=False)[0]
            except Exception as e:
                logger.warning(f"Preprocessor failed in batch explain flow: {e}")
                # fallback: try to scale using scale_features (fit=False) or use raw
                try:
                    X = preprocessor.scale_features(X_raw, target_column=None, fit=False)[0]
                except Exception:
                    X = X_raw
        
        # Generate explanations for selected samples
        explanations = {}
        for idx in sample_indices:
            if idx < len(X):
                sample_explanations = {}
                
                # SHAP explanation
                shap_expl = explainer.explain_with_shap(X, sample_idx=idx)
                if shap_expl:
                    sample_explanations['shap'] = {
                        'shap_values': shap_expl['shap_values'].tolist() if 'shap_values' in shap_expl else None,
                        'type': shap_expl.get('type', 'unknown')
                    }
                
                # LIME explanation
                lime_expl = explainer.explain_with_lime(X, sample_idx=idx)
                if lime_expl:
                    sample_explanations['lime'] = {
                        'feature_importance': lime_expl.get('feature_importance', []),
                        'sample_idx': lime_expl.get('sample_idx', idx)
                    }
                
                explanations[idx] = sample_explanations
        
        return jsonify({
            'explanations': explanations,
            'n_samples_explained': len(explanations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch explanation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new model (for development/testing purposes)."""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'ml')  # 'ml' or 'dl'
        model_name = data.get('model_name', 'random_forest')
        
        # Load and preprocess data
        data_file = os.path.join(DATA_DIR, 'parkinsons_disease_data.csv')
        if not os.path.exists(data_file):
            return jsonify({'error': 'Training data not found'}), 400
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocessor.preprocess_pipeline(data_file)
        
        if model_type == 'ml':
            # Train ML model
            trainer = MLModelTrainer()
            model, training_time = trainer.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            metrics = trainer.evaluate_model(model_name, X_test, y_test)
            
            # Save model (use .joblib to match loader expectations)
            model_path = os.path.join(MODELS_DIR, f'{model_name}_model.joblib')
            trainer.save_model(model_name, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
            
        elif model_type == 'dl':
            # Train DNN model
            trainer = DLModelTrainer()
            model = trainer.train_dnn_model(model_name, X_train.values, y_train.values)
            
            # Evaluate model
            metrics = trainer.evaluate_dnn_model(model_name, X_test.values, y_test.values)
            
            # Save model
            # Save DNN model using a consistent filename for loader
            model_path = os.path.join(MODELS_DIR, 'dnn_model.keras')
            trainer.save_model(model_name, model_path)
            
            # Save preprocessor
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
        
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        # Reload models
        # Reload models into the global registry and (re)initialize explainer
        load_models()
        initialize_explainer()

        return jsonify({
            'message': f'Model {model_name} trained successfully',
            'model_path': model_path,
            'metrics': metrics,
            'training_time': training_time if model_type == 'ml' else 'N/A'
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

def save_model_results(model_name, results, output_dir):
    """Save model results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model results saved to {filepath}")
    return filepath

@app.route('/api/reports/status')
def report_status():
    """Get status of generated reports."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_dir):
        return jsonify({
            'status': 'missing',
            'message': 'No reports generated yet'
        })
    
    files = []
    total_size = 0
    for f in os.listdir(results_dir):
        file_path = os.path.join(results_dir, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            files.append({
                'name': f,
                'size': size,
                'created': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                'type': os.path.splitext(f)[1][1:]  # file extension without dot
            })
            total_size += size
    
    return jsonify({
        'status': 'ready',
        'count': len(files),
        'total_size': total_size,
        'files': files,
        'last_generated': max([f['created'] for f in files]) if files else None
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    initialize_explainer()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
