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

# Figure styling for high-quality output
sns.set_theme(style='whitegrid')
sns.set_context('talk', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 6)
})


def _save_figure(base_name, results_dir='results', dpi=300, transparent=False):
    """Save the current matplotlib figure as a high-quality JPEG.

    File written: results/{base_name}.jpg
    """
    os.makedirs(results_dir, exist_ok=True)
    jpg_path = os.path.join(results_dir, f"{base_name}.jpg")
    png_fallback = os.path.join(results_dir, f"{base_name}.png")
    try:
        # Save JPEG (matplotlib uses Pillow for JPEG output)
        plt.savefig(jpg_path, dpi=dpi, bbox_inches='tight', transparent=transparent, quality=95)
    except Exception:
        # If JPEG saving fails for any reason, fall back to PNG
        try:
            plt.savefig(png_fallback, dpi=dpi, bbox_inches='tight')
        except Exception:
            # Last resort: try the default save without extra options
            plt.savefig(png_fallback)


def load_data():
    """Load and prepare the dataset (apply same column drops as training)."""
    print("Loading dataset...")
    data = pd.read_csv("data/parkinsons_disease_data.csv")

    # Drop metadata columns used in training
    for c in ["PatientID", "DoctorInCharge"]:
        if c in data.columns:
            data = data.drop(columns=[c])

    X_all = data.drop(["Diagnosis"], axis=1)
    y_all = data["Diagnosis"]

    # If training saved test indices, use them for reproducible plots
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

    # Apply saved preprocessor to align/scale features if available
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
    """Load models from the models/ directory.
    Missing optional models are skipped with a message.
    """
    print("Loading models...")
    models = {}
    try:
        models['random_forest'] = joblib.load('models/random_forest_model.joblib')
    except Exception:
        print("random_forest model not found")
    try:
        models['xgboost'] = joblib.load('models/xgboost_model.joblib')
    except Exception:
        print("xgboost model not found")
    try:
        models['svm'] = joblib.load('models/svm_model.joblib')
    except Exception:
        print("svm model not found")
    try:
        models['logistic'] = joblib.load('models/logistic_model.joblib')
    except Exception:
        print("logistic model not found")

    # DNN is optional
    if tf is not None:
        try:
            models['dnn'] = tf.keras.models.load_model('models/dnn_model.keras')
        except Exception:
            print("DNN model could not be loaded")

    return models


def plot_model_performance(models, X, y):
    """Create high-resolution model accuracy comparison bar plot."""
    print("Generating performance comparison plot...")
    results = {}

    for name, model in models.items():
        if name == 'dnn':
            continue
        try:
            y_pred = model.predict(X)
        except Exception:
            # some models return probabilities by default
            y_pred = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
        accuracy = (y_pred == y).mean()
        results[name] = accuracy

    labels = list(results.keys())
    accuracies = [results[k] for k in labels]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=labels, y=accuracies, palette='muted')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=12)

    _save_figure('model_comparison')
    plt.close()


def plot_confusion_matrices(models, X, y):
    """Generate confusion matrices for available models and save as a high-res figure."""
    print("Generating confusion matrices...")
    available = [k for k in models.keys() if k != 'dnn']
    n = len(available)
    cols = min(3, max(1, n))
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, name in enumerate(available):
        model = models[name]
        try:
            y_pred = model.predict(X)
        except Exception:
            y_pred = (model.predict_proba(X)[:, 1] > 0.5).astype(int)
        cm = confusion_matrix(y, y_pred)
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', annot_kws={"fontsize":12})
        ax.set_title(f'{name.capitalize()} Confusion Matrix', fontsize=14)

    # hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)
    _save_figure('confusion_matrices')
    plt.close()


def plot_roc_curves(models, X, y):
    """Generate ROC curves and save high-resolution plot."""
    print("Generating ROC curves...")
    plt.figure(figsize=(12, 8))

    for name, model in models.items():
        if name == 'dnn':
            continue
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                y_pred_proba = model.decision_function(X)
        except Exception:
            continue
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    _save_figure('roc_curves')
    plt.close()


def generate_feature_importance(models, X):
    """Generate feature importance barplots for tree-based models."""
    print("Generating feature importance plots...")
    feature_names = X.columns

    if 'random_forest' in models:
        rf = models['random_forest']
        rf_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=rf_importance.head(15), x='importance', y='feature', palette='crest')
        plt.title('Random Forest Feature Importance (Top 15)')
        _save_figure('rf_feature_importance')
        plt.close()

    if 'xgboost' in models:
        xgb = models['xgboost']
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(data=xgb_importance.head(15), x='importance', y='feature', palette='mako')
        plt.title('XGBoost Feature Importance (Top 15)')
        _save_figure('xgb_feature_importance')
        plt.close()


def generate_shap_plots(models, X):
    """Generate SHAP summary plots (bar and full) for tree models (XGBoost).

    These are saved as high-res PNG and vector formats.
    """
    if 'xgboost' not in models:
        print('XGBoost model not available — skipping SHAP plots')
        return

    print("Generating SHAP plots...")
    explainer = shap.TreeExplainer(models['xgboost'])
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (XGBoost)')
    _save_figure('shap_importance')
    plt.close()

    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot')
    _save_figure('shap_summary')
    plt.close()


def main():
    os.makedirs('results', exist_ok=True)
    X, y = load_data()
    models = load_models()

    plot_model_performance(models, X, y)
    plot_confusion_matrices(models, X, y)
    plot_roc_curves(models, X, y)
    generate_feature_importance(models, X)
    generate_shap_plots(models, X)

    print("Results generation completed successfully!")


if __name__ == "__main__":
    main()