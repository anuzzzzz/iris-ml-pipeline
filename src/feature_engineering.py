import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
import os

def engineer_features(input_path, output_path):
    """Engineer features for IRIS dataset"""
    
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Create unique entity IDs
    df['entity_id'] = df.index.astype(str)
    
    # Feature engineering
    df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
    df['total_area'] = (df['sepal length (cm)'] * df['sepal width (cm)'] + 
                       df['petal length (cm)'] * df['petal width (cm)'])
    
    # Add timestamp features
    base_time = datetime.now()
    df['event_timestamp'] = [base_time - timedelta(hours=i) for i in range(len(df))]
    df['created_timestamp'] = datetime.now()
    
    # Rename columns for feature store
    feature_df = df.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    })
    
    # Select features for feature store
    features = [
        'entity_id', 'sepal_length', 'sepal_width', 'petal_length', 
        'petal_width', 'sepal_ratio', 'petal_ratio', 'total_area',
        'event_timestamp', 'created_timestamp'
    ]
    
    feature_df = feature_df[features]
    
    # Save processed features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    
    print(f"Engineered features saved to {output_path}")
    print(f"Feature shape: {feature_df.shape}")
    print(f"Features: {feature_df.columns.tolist()}")
    
    return feature_df

def load_features_to_bigquery(csv_path, project_id, dataset_id, table_id):
    """Load features to BigQuery"""
    
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    # Configure load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition="WRITE_TRUNCATE"
    )
    
    # Load data
    with open(csv_path, "rb") as source_file:
        job = client.load_table_from_file(
            source_file, table_ref, job_config=job_config
        )
    
    job.result()  # Wait for job to complete
    
    table = client.get_table(table_ref)
    print(f"Loaded {table.num_rows} rows to {table_ref}")

if __name__ == "__main__":
    # Engineer features
    feature_df = engineer_features('data/raw/iris.csv', 'data/features/iris_features.csv')
    
    # Load to BigQuery
    project_id = os.getenv('PROJECT_ID', 'your-gcp-project-id')
    load_features_to_bigquery(
        'data/features/iris_features.csv',
        project_id,
        'iris_feature_store',
        'iris_features'
    )
