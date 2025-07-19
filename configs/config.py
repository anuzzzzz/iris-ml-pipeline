import os

# GCP Configuration
PROJECT_ID = os.getenv('PROJECT_ID', 'elaborate-chess-461609-j9')
BUCKET_NAME = os.getenv('BUCKET_NAME', 'iris-ml-pipeline-dvc-elaborate-chess-461609-j9')
REGION = os.getenv('REGION', 'us-central1')

# Data paths
DATA_RAW_PATH = 'data/raw'
DATA_PROCESSED_PATH = 'data/processed'
MODELS_PATH = 'models'

# Model configuration
MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 5,
    'random_state': 42
}

# DVC Remote
DVC_REMOTE = 'gs://iris-ml-pipeline-dvc-elaborate-chess-461609-j9/dvc-store'
