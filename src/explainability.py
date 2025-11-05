"""
Explainability module for Parkinson's Disease Prediction using XAI.
Implements SHAP, LIME, and other explainability techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import shap
    shap_available = True
except Exception as _:
    shap = None
    shap_available = False

try:
    import lime
    import lime.lime_tabular
    lime_available = True
except Exception as _:
    lime = None
    lime_available = False
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Class for explaining model predictions using various XAI techniques."""
    
    def __init__(self, model, feature_names=None, class_names=None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model to explain
            feature_names (list): List of feature names
            class_names (list): List of class names
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['No Parkinson', 'Parkinson']
        self.explanations = {}
        
    def set_feature_names(self, feature_names):
        """Set feature names for explanations."""
        self.feature_names = feature_names
        logger.info(f"Feature names set: {len(feature_names)} features")
    
    def explain_with_shap(self, X, sample_idx=None, max_display=20):
        """
        Explain model predictions using SHAP.
        
        Args:
            X (pd.DataFrame): Feature matrix
            sample_idx (int): Index of specific sample to explain (None for all)
            max_display (int): Maximum number of features to display
            
        Returns:
            dict: SHAP explanations
        """
        if not shap_available:
            logger.error("SHAP is not available in the environment. Install shap to use this feature.")
            return None

        try:
            logger.info("Generating SHAP explanations...")
            
            # Initialize SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict_proba, X)
            else:
                explainer = shap.TreeExplainer(self.model) if hasattr(self.model, 'estimators_') else shap.KernelExplainer(self.model.predict, X)
            
            # Generate SHAP values
            if sample_idx is not None:
                # Explain specific sample
                sample = X.iloc[sample_idx:sample_idx+1]
                if hasattr(explainer, 'shap_values'):
                    shap_values = explainer.shap_values(sample)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification, use positive class
                else:
                    shap_values = explainer.shap_values(sample)
                
                # Create explanation for single sample
                explanation = {
                    'shap_values': shap_values,
                    'sample': sample,
                    'type': 'single_sample'
                }
                
                # Plot waterfall plot for single sample
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                   base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                                   data=sample.iloc[0].values,
                                                   feature_names=self.feature_names),
                                  max_display=max_display)
                plt.title(f'SHAP Explanation for Sample {sample_idx}')
                plt.tight_layout()
                plt.show()
                
            else:
                # Explain all samples
                if hasattr(explainer, 'shap_values'):
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification, use positive class
                else:
                    shap_values = explainer.shap_values(X)
                
                explanation = {
                    'shap_values': shap_values,
                    'type': 'all_samples'
                }
                
                # Plot summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X, feature_names=self.feature_names, max_display=max_display)
                plt.title('SHAP Feature Importance Summary')
                plt.tight_layout()
                plt.show()
                
                # Plot bar plot
                plt.figure(figsize=(10, 8))
                shap.plots.bar(shap_values, max_display=max_display)
                plt.title('SHAP Feature Importance (Bar Plot)')
                plt.tight_layout()
                plt.show()
            
            self.explanations['shap'] = explanation
            logger.info("SHAP explanations generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return None
    
    def explain_with_lime(self, X, sample_idx, num_features=10):
        """
        Explain model predictions using LIME.
        
        Args:
            X (pd.DataFrame): Feature matrix
            sample_idx (int): Index of sample to explain
            num_features (int): Number of top features to show
            
        Returns:
            dict: LIME explanation
        """
        if not lime_available:
            logger.error("LIME is not available in the environment. Install lime to use this feature.")
            return None

        try:
            logger.info(f"Generating LIME explanation for sample {sample_idx}...")
            
            # Prepare data for LIME
            X_array = X.values
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_array,
                feature_names=feature_names,
                class_names=self.class_names,
                mode='classification'
            )
            
            # Explain the specific sample
            sample = X.iloc[sample_idx].values
            explanation = explainer.explain_instance(
                sample,
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=num_features
            )
            
            # Store explanation
            lime_explanation = {
                'explanation': explanation,
                'sample_idx': sample_idx,
                'sample': sample,
                'feature_importance': explanation.as_list()
            }
            
            # Plot LIME explanation
            plt.figure(figsize=(10, 6))
            explanation.as_pyplot_figure()
            plt.title(f'LIME Explanation for Sample {sample_idx}')
            plt.tight_layout()
            plt.show()
            
            self.explanations['lime'] = lime_explanation
            logger.info("LIME explanation generated successfully")
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            return None
    
    def explain_feature_importance(self, X, y, method='shap', top_n=20):
        """
        Explain feature importance using various methods.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            method (str): Method to use ('shap', 'permutation', 'correlation')
            top_n (int): Number of top features to display
            
        Returns:
            dict: Feature importance explanation
        """
        if method == 'shap':
            return self._explain_shap_importance(X, top_n)
        elif method == 'permutation':
            return self._explain_permutation_importance(X, y, top_n)
        elif method == 'correlation':
            return self._explain_correlation_importance(X, y, top_n)
        else:
            raise ValueError("method must be 'shap', 'permutation', or 'correlation'")
    
    def _explain_shap_importance(self, X, top_n=20):
        """Explain feature importance using SHAP values."""
        try:
            # Generate SHAP explanations first
            if 'shap' not in self.explanations:
                self.explain_with_shap(X)
            
            if 'shap' in self.explanations:
                shap_values = self.explanations['shap']['shap_values']
                
                # Calculate mean absolute SHAP values
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': mean_shap
                }).sort_values('importance', ascending=False).head(top_n)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(data=importance_df, x='importance', y='feature')
                plt.title(f'Top {top_n} Features by SHAP Importance')
                plt.xlabel('Mean |SHAP Value|')
                plt.ylabel('Features')
                plt.tight_layout()
                plt.show()
                
                return importance_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error explaining SHAP importance: {e}")
            return None
    
    def _explain_permutation_importance(self, X, y, top_n=20):
        """Explain feature importance using permutation importance."""
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance
            result = permutation_importance(
                self.model, X, y, 
                n_repeats=10, 
                random_state=42,
                n_jobs=-1
            )
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Features by Permutation Importance')
            plt.xlabel('Permutation Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
            
            return importance_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error explaining permutation importance: {e}")
            return None
    
    def _explain_correlation_importance(self, X, y, top_n=20):
        """Explain feature importance using correlation with target."""
        try:
            # Calculate correlation with target
            correlations = []
            for feature in X.columns:
                corr = np.corrcoef(X[feature], y)[0, 1]
                correlations.append(abs(corr))
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'correlation': correlations
            }).sort_values('correlation', ascending=False).head(top_n)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='correlation', y='feature')
            plt.title(f'Top {top_n} Features by Absolute Correlation with Target')
            plt.xlabel('|Correlation|')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
            
            return importance_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error explaining correlation importance: {e}")
            return None
    
    def explain_prediction_confidence(self, X, sample_idx):
        """
        Explain prediction confidence for a specific sample.
        
        Args:
            X (pd.DataFrame): Feature matrix
            sample_idx (int): Index of sample to explain
            
        Returns:
            dict: Confidence explanation
        """
        try:
            sample = X.iloc[sample_idx:sample_idx+1]
            
            # Get prediction and confidence
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(sample)[0]
                prediction = self.model.predict(sample)[0]
                confidence = max(proba)
            else:
                prediction = self.model.predict(sample)[0]
                confidence = 1.0  # Default confidence for models without probability
                proba = [1.0, 0.0] if prediction == 1 else [0.0, 1.0]
            
            # Get feature contributions (using SHAP if available)
            feature_contributions = None
            if 'shap' in self.explanations and self.explanations['shap']['type'] == 'single_sample':
                shap_values = self.explanations['shap']['shap_values']
                feature_contributions = dict(zip(self.feature_names, shap_values[0]))
            
            explanation = {
                'sample_idx': sample_idx,
                'prediction': int(prediction),
                'prediction_class': self.class_names[int(prediction)],
                'confidence': confidence,
                'probabilities': {
                    'No Parkinson': proba[0],
                    'Parkinson': proba[1]
                },
                'feature_contributions': feature_contributions
            }
            
            # Plot confidence visualization
            plt.figure(figsize=(10, 6))
            
            # Probability bar plot
            plt.subplot(1, 2, 1)
            classes = ['No Parkinson', 'Parkinson']
            colors = ['lightblue', 'lightcoral']
            bars = plt.bar(classes, proba, color=colors)
            plt.title('Prediction Probabilities')
            plt.ylabel('Probability')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, proba):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom')
            
            # Confidence indicator
            plt.subplot(1, 2, 2)
            plt.pie([confidence, 1-confidence], labels=['Confidence', 'Uncertainty'], 
                   colors=['lightgreen', 'lightgray'], autopct='%1.1f%%')
            plt.title('Prediction Confidence')
            
            plt.suptitle(f'Prediction Explanation for Sample {sample_idx}')
            plt.tight_layout()
            plt.show()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction confidence: {e}")
            return None
    
    def generate_explanation_report(self, X, y, sample_indices=None, output_file=None):
        """
        Generate a comprehensive explanation report.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            sample_indices (list): Indices of samples to explain in detail
            output_file (str): Path to save the report
        """
        try:
            logger.info("Generating comprehensive explanation report...")
            
            report = {
                'model_type': type(self.model).__name__,
                'dataset_info': {
                    'n_samples': len(X),
                    'n_features': len(X.columns),
                    'feature_names': self.feature_names,
                    'class_distribution': y.value_counts().to_dict()
                },
                'explanations': {}
            }
            
            # Generate SHAP explanations
            report['explanations']['shap'] = self.explain_with_shap(X)
            
            # Generate feature importance explanations
            report['explanations']['feature_importance'] = {
                'shap': self._explain_shap_importance(X),
                'permutation': self._explain_permutation_importance(X, y),
                'correlation': self._explain_correlation_importance(X, y)
            }
            
            # Generate detailed explanations for specific samples
            if sample_indices:
                report['explanations']['sample_explanations'] = {}
                for idx in sample_indices:
                    if idx < len(X):
                        report['explanations']['sample_explanations'][idx] = {
                            'lime': self.explain_with_lime(X, idx),
                            'confidence': self.explain_prediction_confidence(X, idx)
                        }
            
            # Save report if file path provided
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Explanation report saved to {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return None

def main():
    """Example usage of the ModelExplainer class."""
    print("ModelExplainer class created successfully!")
    print("\nThis class provides:")
    print("1. SHAP explanations for global and local interpretability")
    print("2. LIME explanations for individual predictions")
    print("3. Feature importance analysis using multiple methods")
    print("4. Prediction confidence explanations")
    print("5. Comprehensive explanation reports")
    print("\nTo use this class:")
    print("1. Train a model first")
    print("2. Create a ModelExplainer instance with the trained model")
    print("3. Call various explanation methods")
    print("4. Generate comprehensive reports for stakeholders")

if __name__ == "__main__":
    main()
