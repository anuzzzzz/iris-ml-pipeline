import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def preprocess_data(input_path, output_dir):
    """Preprocess IRIS data for training"""
    
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Prepare features and target
    X = df[['sepal length (cm)', 'sepal width (cm)', 
            'petal length (cm)', 'petal width (cm)']]
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        f'{output_dir}/X_train.csv', index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        f'{output_dir}/X_test.csv', index=False
    )
    pd.DataFrame(y_train).to_csv(f'{output_dir}/y_train.csv', index=False)
    pd.DataFrame(y_test).to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    print(f"Processed data saved to {output_dir}")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    preprocess_data('data/raw/iris.csv', 'data/processed')
