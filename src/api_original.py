from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class IrisPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_names = ['setosa', 'versicolor', 'virginica']
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Load model from your existing models directory
            model_path = os.path.join('models', 'iris_model.pkl')
            scaler_path = os.path.join('data', 'processed', 'scaler.pkl')
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler not found at {scaler_path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict(self, features):
        """Make prediction on input features"""
        try:
            # Convert to numpy array and reshape
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler is available
            if self.scaler:
                features_array = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_array)[0]
            probabilities = self.model.predict_proba(features_array)[0]
            
            return {
                'prediction': int(prediction),
                'class_name': self.class_names[prediction],
                'probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                'confidence': float(max(probabilities))
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise e

# Initialize predictor
predictor = IrisPredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'iris-api',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor.model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'IRIS Classification API',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'predict': '/predict',
            'batch_predict': '/batch_predict',
            'model_info': '/model_info'
        },
        'usage': {
            'predict': 'POST /predict with {"features": [5.1, 3.5, 1.4, 0.2]}',
            'batch_predict': 'POST /batch_predict with {"features": [[5.1, 3.5, 1.4, 0.2], ...]}'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features in request body',
                'expected_format': {'features': [5.1, 3.5, 1.4, 0.2]}
            }), 400
        
        features = data['features']
        
        # Validate features
        if len(features) != 4:
            return jsonify({
                'error': 'Expected 4 features (sepal_length, sepal_width, petal_length, petal_width)',
                'received': len(features)
            }), 400
        
        # Make prediction
        result = predictor.predict(features)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'input_features': {
                'sepal_length': features[0],
                'sepal_width': features[1],
                'petal_length': features[2],
                'petal_width': features[3]
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Missing features in request body',
                'expected_format': {'features': [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}
            }), 400
        
        features_list = data['features']
        predictions = []
        
        for i, features in enumerate(features_list):
            if len(features) != 4:
                return jsonify({
                    'error': f'Sample {i}: Expected 4 features, got {len(features)}'
                }), 400
            
            result = predictor.predict(features)
            predictions.append({
                'sample_id': i,
                'prediction': result,
                'input_features': features
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_samples': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    try:
        # Load model metrics if available
        metrics_path = os.path.join('models', 'metrics.json')
        metrics = {}
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return jsonify({
            'model_type': 'Iris Classification',
            'features': [
                'sepal_length',
                'sepal_width', 
                'petal_length',
                'petal_width'
            ],
            'classes': predictor.class_names,
            'metrics': metrics,
            'model_loaded': predictor.model is not None,
            'scaler_loaded': predictor.scaler is not None
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/predict', '/batch_predict', '/model_info']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please check the logs for more details'
    }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
