import pytest
import pandas as pd
import os
import joblib
from src.preprocess import preprocess_data
from src.train import train_model

def test_data_exists():
    """Test if raw data exists"""
    assert os.path.exists('data/raw/iris.csv'), "Raw data file missing"
    df = pd.read_csv('data/raw/iris.csv')
    assert df.shape[0] == 150, "IRIS dataset should have 150 samples"
    assert df.shape[1] == 6, "IRIS dataset should have 6 columns"

def test_processed_data_exists():
    """Test if processed data exists"""
    assert os.path.exists('data/processed/X_train.csv'), "Processed training data missing"
    assert os.path.exists('data/processed/X_test.csv'), "Processed test data missing"
    assert os.path.exists('data/processed/scaler.pkl'), "Scaler file missing"

def test_model_exists():
    """Test if model exists"""
    assert os.path.exists('models/iris_model.pkl'), "Model file missing"
    assert os.path.exists('models/metrics.json'), "Metrics file missing"

def test_model_performance():
    """Test model performance"""
    import json
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    assert metrics['accuracy'] > 0.8, "Model accuracy should be > 80%"

if __name__ == "__main__":
    pytest.main([__file__])
