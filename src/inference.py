import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os

class IrisPredictor:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.target_names = ['setosa', 'versicolor', 'virginica']
    
    def predict(self, features):
        """Make prediction on new data"""
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif isinstance(features, list):
            features = pd.DataFrame(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(prediction, probability)):
            result = {
                'prediction': int(pred),
                'prediction_label': self.target_names[pred],
                'probability': prob.tolist(),
                'confidence': float(max(prob)),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        return results if len(results) > 1 else results[0]
    
    def batch_predict(self, csv_path, output_path):
        """Batch prediction from CSV file"""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Extract features
        feature_columns = ['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)']
        features = df[feature_columns]
        
        # Make predictions
        predictions = []
        for _, row in features.iterrows():
            pred_result = self.predict(row.to_dict())
            predictions.append(pred_result)
        
        # Save results
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(output_path, index=False)
        
        print(f"Batch predictions saved to {output_path}")
        return results_df

def main():
    """CLI interface for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='IRIS Model Inference')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--scaler', required=True, help='Path to scaler file')
    parser.add_argument('--input', required=True, help='Input CSV file or JSON string')
    parser.add_argument('--output', help='Output CSV file (for batch prediction)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = IrisPredictor(args.model, args.scaler)
    
    # Check if input is CSV file or JSON
    if args.input.endswith('.csv'):
        # Batch prediction
        if not args.output:
            args.output = args.input.replace('.csv', '_predictions.csv')
        
        results = predictor.batch_predict(args.input, args.output)
        print(f"Processed {len(results)} samples")
    else:
        # Single prediction from JSON
        try:
            features = json.loads(args.input)
            result = predictor.predict(features)
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError:
            print("Invalid JSON input")

if __name__ == "__main__":
    main()
