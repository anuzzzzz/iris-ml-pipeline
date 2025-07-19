import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import os

def prepare_iris_data():
    """Prepare IRIS dataset and save to CSV"""
    
    # Load IRIS dataset
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map(dict(enumerate(iris.target_names)))
    
    # Add timestamp (for later feature store usage)
    df['timestamp'] = pd.Timestamp.now()
    
    # Ensure data directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Save raw data
    df.to_csv('data/raw/iris.csv', index=False)
    print(f"Dataset saved to data/raw/iris.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    df = prepare_iris_data()
    print(df.head())
