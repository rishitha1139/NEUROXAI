"""
Data preprocessing module for Parkinson's Disease Prediction using XAI.
Handles data cleaning, feature scaling, and data splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing Parkinson's disease dataset."""
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type (str): Type of scaler to use ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, file_path):
        """
        Load the Parkinson's disease dataset.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def check_data_quality(self, data):
        """
        Check data quality and provide summary statistics.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            dict: Data quality summary
        """
        quality_report = {
            'shape': data.shape,
            'missing_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'data_types': data.dtypes.to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': data.select_dtypes(include=['object']).columns.tolist()
        }
        
        logger.info("Data quality report generated")
        return quality_report
    
    def handle_missing_values(self, data, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            pd.DataFrame: Dataset with imputed values
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if data[numeric_cols].isnull().any().any():
            logger.info(f"Handling missing values using {strategy} strategy")
            self.imputer = SimpleImputer(strategy=strategy)
            data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
            logger.info("Missing values handled successfully")
        else:
            logger.info("No missing values found in numeric columns")
            
        return data
    
    def scale_features(self, data, target_column='status'):
        """
        Scale numerical features.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_column (str): Name of the target variable
            
        Returns:
            tuple: (X_scaled, y, scaler)
        """
        # Separate features and target
        if target_column in data.columns:
            X = data.drop(columns=[target_column])
            y = data[target_column]
        else:
            X = data
            y = None
            
        # Select only numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_features]
        
        # Initialize and fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        X_scaled = self.scaler.fit_transform(X_numeric)
        X_scaled = pd.DataFrame(X_scaled, columns=numeric_features, index=X.index)
        
        # Add back non-numeric features if any
        non_numeric_features = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_features) > 0:
            X_scaled = pd.concat([X_scaled, X[non_numeric_features]], axis=1)
        
        logger.info(f"Features scaled using {self.scaler_type} scaler")
        return X_scaled, y, self.scaler
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path, target_column='status', test_size=0.2):
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to the CSV file
            target_column (str): Name of the target variable
            test_size (float): Proportion of test set
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler, feature_names)
        """
        # Load data
        data = self.load_data(file_path)
        
        # Check data quality
        quality_report = self.check_data_quality(data)
        logger.info(f"Data quality report: {quality_report}")
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Scale features
        X_scaled, y, scaler = self.scale_features(data, target_column)
        
        # Store feature names
        self.feature_names = X_scaled.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X_scaled, y, test_size=test_size
        )
        
        logger.info("Preprocessing pipeline completed successfully")
        return X_train, X_test, y_train, y_test, scaler, self.feature_names


        logger.info(f"Data split: Train shape {X_train.shape}, Test shape {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def transform_for_prediction(self, data):
        """
        Prepares new, raw data for prediction using the fitted components.

        Args:
            data (pd.DataFrame): The new data to transform.

        Returns:
            pd.DataFrame: The transformed data, ready for prediction.
        """
        if not self.scaler or not self.feature_names:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Please run the training pipeline first to fit the preprocessor."
            )

        # Handle missing values using the fitted imputer
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if self.imputer and len(numeric_cols) > 0:
            data[numeric_cols] = self.imputer.transform(data[numeric_cols])
        
        # Identify numeric features that were present during training
        numeric_features_to_scale = [
            col for col in self.feature_names 
            if col in data.columns and np.issubdtype(data[col].dtype, np.number)
        ]
        
        if numeric_features_to_scale:
            # Scale numeric features using the fitted scaler
            data_scaled_array = self.scaler.transform(data[numeric_features_to_scale])
            data_scaled = pd.DataFrame(data_scaled_array, columns=numeric_features_to_scale, index=data.index)

            # Update the original dataframe with scaled values
            data[numeric_features_to_scale] = data_scaled

        # Align columns to match the exact feature set from training
        # This adds missing columns with 0 and removes extra ones.
        data_aligned = data.reindex(columns=self.feature_names, fill_value=0)
        
        logger.info("Prediction data transformed successfully.")
        return data_aligned

def main():
    """Example usage of the DataPreprocessor class."""
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    # Example preprocessing pipeline
    try:
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocessor.preprocess_pipeline(
            file_path='../data/parkinsons_disease_data.csv'
        )
        print("Preprocessing completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        print(f"Number of features: {len(feature_names)}")
        
    except FileNotFoundError:
        print("Dataset file not found. Please ensure the CSV file is in the data/ directory.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()
