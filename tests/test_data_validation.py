import pytest
import pandas as pd
import numpy as np
import os
import json
from src.feature_engineering import engineer_features

class TestDataValidation:
    
    def test_raw_data_schema(self):
        """Test raw data schema and quality"""
        df = pd.read_csv('data/raw/iris.csv')
        
        # Check schema
        expected_columns = ['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)', 'target']
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check data quality
        assert df.shape[0] == 150, "Should have 150 samples"
        assert df.isnull().sum().sum() == 0, "Should have no null values"
        assert df['target'].nunique() == 3, "Should have 3 target classes"
    
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        if not os.path.exists('data/features'):
            os.makedirs('data/features')
            
        # Run feature engineering
        feature_df = engineer_features('data/raw/iris.csv', 'data/features/test_features.csv')
        
        # Validate engineered features
        assert 'sepal_ratio' in feature_df.columns, "Missing sepal_ratio feature"
        assert 'petal_ratio' in feature_df.columns, "Missing petal_ratio feature"
        assert 'total_area' in feature_df.columns, "Missing total_area feature"
        
        # Check for invalid values
        assert not feature_df['sepal_ratio'].isin([np.inf, -np.inf]).any(), "Invalid sepal_ratio values"
        assert not feature_df['petal_ratio'].isin([np.inf, -np.inf]).any(), "Invalid petal_ratio values"
        
        # Clean up
        if os.path.exists('data/features/test_features.csv'):
            os.remove('data/features/test_features.csv')
    
    def test_model_performance(self):
        """Test model performance thresholds"""
        if os.path.exists('models/metrics_mlflow.json'):
            with open('models/metrics_mlflow.json', 'r') as f:
                metrics = json.load(f)
            
            assert metrics['accuracy'] > 0.85, f"Accuracy too low: {metrics['accuracy']}"
            assert metrics['cv_score'] > 0.80, f"CV score too low: {metrics['cv_score']}"
        else:
            pytest.skip("Model metrics not found")
    
    def test_data_drift(self):
        """Test for data drift (basic implementation)"""
        df = pd.read_csv('data/raw/iris.csv')
        
        # Check feature distributions
        for col in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:
            assert df[col].std() > 0, f"No variation in {col}"
            assert df[col].min() >= 0, f"Negative values in {col}"
    
    def test_processed_data_consistency(self):
        """Test consistency between train and test splits"""
        if os.path.exists('data/processed/X_train.csv') and os.path.exists('data/processed/X_test.csv'):
            X_train = pd.read_csv('data/processed/X_train.csv')
            X_test = pd.read_csv('data/processed/X_test.csv')
            
            # Check columns match
            assert list(X_train.columns) == list(X_test.columns), "Train/test columns mismatch"
            
            # Check for data leakage (distributions should be similar)
            for col in X_train.columns:
                train_mean = X_train[col].mean()
                test_mean = X_test[col].mean()
                assert abs(train_mean - test_mean) < 1.0, f"Large difference in {col} means"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
