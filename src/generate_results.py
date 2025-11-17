import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import shap

# Reproducible seeds
seed = 42
random.seed(seed)
np.random.seed(seed)

try:
    import tensorflow as tf
    tf.random.set_seed(seed)
except Exception:
    tf = None

def load_data():
    """Load and prepare the dataset"""
    print("Loading dataset...")
    data = pd.read_csv("data/parkinsons_disease_data.csv")

    # Drop metadata columns like during training
    for c in ["PatientID", "DoctorInCharge"]:
        if c in data.columns:
            data = data.drop(columns=[c])

    X_all = data.drop(["Diagnosis"], axis=1)
    y_all = data["Diagnosis"]

    # If training saved test indices, load and select the same test set
    test_idx_path = 'models/test_indices.joblib'
    if os.path.exists(test_idx_path):
        try:
            test_idx = joblib.load(test_idx_path)
            X = X_all.loc[test_idx]
            y = y_all.loc[test_idx]
            print(f"Loaded test indices from {test_idx_path}, using {len(test_idx)} samples for evaluation")
        except Exception as e:
            print(f"Could not load test indices ({e}), falling back to full dataset")
            X = X_all
            y = y_all
    else:
        X = X_all
        y = y_all

    # If a preprocessor exists, use it to align/scale features
    preprocessor_path = 'models/preprocessor.pkl'
    if os.path.exists(preprocessor_path):
        try:
            preprocessor = joblib.load(preprocessor_path)
            X = preprocessor.prepare_inference(X.copy())
            print("Applied saved preprocessor to evaluation dataset")
        except Exception as e:
            print(f"Could not apply preprocessor: {e}")

    return X, y

def load_models():
    """Load all trained models"""
    print("Loading models...")
    models = {
        'random_forest': joblib.load('models/random_forest_model.joblib'),
        'xgboost': joblib.load('models/xgboost_model.joblib'),
        'svm': joblib.load('models/svm_model.joblib'),
        'logistic': joblib.load('models/logistic_model.joblib')
    }
    try:
        models['dnn'] = tf.keras.models.load_model('models/dnn_model.keras')
    except:
        print("DNN model could not be loaded")
    return models

def plot_model_performance(models, X, y):
    """Create performance comparison plot"""
    print("Generating performance comparison plot...")
    results = {}
    
    for name, model in models.items():
        if name != 'dnn':
            y_pred = model.predict(X)
            accuracy = (y_pred == y).mean()
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y, y_pred, output_dict=True)
            }
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    accuracies = [results[model]['accuracy'] for model in results.keys()]
    plt.bar(results.keys(), accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.savefig('results/model_comparison.png')
    plt.close()

def plot_confusion_matrices(models, X, y):
    """Generate confusion matrices for all models"""
    print("Generating confusion matrices...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        if name != 'dnn':
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'{name.capitalize()} Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')
    plt.close()

def plot_roc_curves(models, X, y):
    """Generate ROC curves for all models"""
    print("Generating ROC curves...")
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        if name != 'dnn':
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                y_pred_proba = model.decision_function(X)
            
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curves.png')
    plt.close()

def generate_feature_importance(models, X):
    """Generate feature importance plots"""
    print("Generating feature importance plots...")
    feature_names = X.columns
    
    # Random Forest feature importance
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['random_forest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rf_importance.head(15), x='importance', y='feature')
    plt.title('Random Forest Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig('results/rf_feature_importance.png')
    plt.close()
    
    # XGBoost feature importance
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': models['xgboost'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=xgb_importance.head(15), x='importance', y='feature')
    plt.title('XGBoost Feature Importance (Top 15)')
    plt.tight_layout()
    plt.savefig('results/xgb_feature_importance.png')
    plt.close()

def generate_shap_plots(models, X):
    """Generate SHAP value plots for model interpretability"""
    print("Generating SHAP plots...")
    # XGBoost SHAP values
    explainer = shap.TreeExplainer(models['xgboost'])
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig('results/shap_importance.png')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot')
    plt.tight_layout()
    plt.savefig('results/shap_summary.png')
    plt.close()

def save_plot(fig, filename, results_dir='results'):
    """Safely save a plot to the results directory."""
    try:
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        fig.savefig(filepath)
        print(f"Saved plot to {filepath}")
    except Exception as e:
        print(f"Failed to save plot {filename}: {e}")
def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load data and models
    X, y = load_data()
    models = load_models()
    
    # Generate various plots and results
    plot_model_performance(models, X, y)
    plot_confusion_matrices(models, X, y)
    plot_roc_curves(models, X, y)
    generate_feature_importance(models, X)
    generate_shap_plots(models, X)
    
    print("Results generation completed successfully!")

if __name__ == "__main__":
    main()