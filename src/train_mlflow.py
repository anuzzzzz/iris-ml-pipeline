import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train_model_with_mlflow(data_dir, model_dir, experiment_name="iris_classification"):
    """Train IRIS model with MLflow tracking and hyperparameter tuning"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Load processed data
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv')['target']
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')['target']
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Start MLflow run
    with mlflow.start_run():
        
        # Create base model
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("cv_score", grid_search.best_score_)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model locally
        joblib.dump(best_model, f'{model_dir}/iris_model_mlflow.pkl')
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'cv_score': grid_search.best_score_,
            'best_params': grid_search.best_params_,
            'model_type': 'RandomForestClassifier'
        }
        
        with open(f'{model_dir}/metrics_mlflow.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Log confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        np.savetxt(f'{model_dir}/confusion_matrix.txt', cm, fmt='%d')
        mlflow.log_artifact(f'{model_dir}/confusion_matrix.txt')
        
        print(f"Model trained and logged to MLflow")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        return best_model, accuracy, grid_search.best_params_

if __name__ == "__main__":
    train_model_with_mlflow('data/processed', 'models')
