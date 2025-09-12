import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, SimpleRNN
import joblib
import os
import sys

def load_data():
    print("Loading dataset from data/parkinsons_disease_data.csv...")
    try:
        data = pd.read_csv("data/parkinsons_disease_data.csv")
        print(f"Dataset loaded successfully. Shape: {data.shape}")
        
        data = data.drop(["PatientID", "DoctorInCharge"], axis=1)
        X = data.drop(["Diagnosis"], axis=1)
        y = data["Diagnosis"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)
        print(f"Data split completed. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_and_save_traditional_models(X_train, X_test, y_train, y_test):
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            gamma=0,
            n_estimators=100
        ),
        'svm': SVC(kernel='rbf', probability=True),
        'logistic': LogisticRegression(random_state=42)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name} accuracy: {score:.4f}")
        
        # Save model
        joblib.dump(model, f"models/{name}_model.joblib")
        print(f"Saved {name} model\n")

def create_dnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_deep_learning_models(X_train, X_test, y_train, y_test):
    try:
        print("\nPreparing data for deep learning model...")
        # Prepare data for deep learning
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
        y_test_cat = tf.keras.utils.to_categorical(y_test, 2)
        print("Data preparation completed successfully")

        # Train DNN model
        print("\nInitializing DNN model...")
        dnn_model = create_dnn_model((X_train.shape[1], 1))
        print("Model architecture created successfully")
        
        print("\nStarting DNN model training...")
        history = dnn_model.fit(
            X_train_reshaped, 
            y_train_cat, 
            epochs=50, 
            batch_size=32, 
            verbose=1,
            validation_data=(X_test_reshaped, y_test_cat)
        )
        
        # Evaluate model
        test_loss, test_accuracy = dnn_model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
        print(f"\nDNN Model Test Accuracy: {test_accuracy:.4f}")
        
        # Save DNN model
        print("\nSaving DNN model...")
        dnn_model.save("models/dnn_model.keras")
        print("DNN model saved successfully\n")
        
    except Exception as e:
        print(f"\nError in deep learning model training: {str(e)}")
        raise

def main():
    try:
        print("\n=== Starting Model Training Pipeline ===\n")
        
        # Create models directory if it doesn't exist
        print("Setting up models directory...")
        os.makedirs("models", exist_ok=True)
        print("Models directory ready\n")

        # Load and split data
        X_train, X_test, y_train, y_test = load_data()

        # Train and save traditional ML models
        print("\n=== Training Traditional ML Models ===")
        train_and_save_traditional_models(X_train, X_test, y_train, y_test)

        # Train and save deep learning models
        print("\n=== Training Deep Learning Models ===")
        train_and_save_deep_learning_models(X_train, X_test, y_train, y_test)
        
        print("\n=== Model Training Pipeline Completed Successfully ===")
        
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
