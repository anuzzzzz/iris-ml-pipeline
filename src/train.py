import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

def train_model(data_dir, model_dir):
    """Train IRIS classification model"""
    
    # Load processed data
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['target']
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')['target']
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, f'{model_dir}/iris_model.pkl')
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 5
    }
    
    with open(f'{model_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model trained and saved to {model_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return model, accuracy

if __name__ == "__main__":
    train_model('data/processed', 'models')
