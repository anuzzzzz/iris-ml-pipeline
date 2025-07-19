import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def evaluate_model(model_path, data_dir, output_dir):
    """Comprehensive model evaluation"""
    
    # Load model and data
    model = joblib.load(model_path)
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')
    y_test = pd.read_csv(f'{data_dir}/y_test.csv')['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Create evaluation report
    evaluation_report = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evaluation report
    with open(f'{output_dir}/evaluation_report.json', 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    # Feature importance plot
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test.columns
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()
    
    print(f"Evaluation complete. Results saved to {output_dir}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return evaluation_report

if __name__ == "__main__":
    evaluate_model('models/iris_model_mlflow.pkl', 'data/processed', 'results')
