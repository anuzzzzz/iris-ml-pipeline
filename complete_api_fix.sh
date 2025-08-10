#!/bin/bash
# complete_api_fix.sh - Add the missing batch prediction endpoint

echo "🔧 Adding missing batch prediction endpoint..."

cd ~/iris-ml-pipline

# Stop the current API
pkill -f "api_port8081.py" 2>/dev/null || true

# Add the batch prediction endpoint to your working API
cat >> src/api_port8081.py << 'BATCH_EOF'

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

BATCH_EOF

echo "✅ Batch prediction endpoint added"

# Restart the API
echo "🚀 Restarting API with batch prediction..."
python src/api_port8081.py > api_8081.log 2>&1 &
NEW_API_PID=$!
echo "🔢 API restarted with PID: $NEW_API_PID"

# Wait for startup
sleep 5

# Test the batch prediction
echo "🧪 Testing batch prediction..."
curl -s -X POST http://127.0.0.1:8081/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8], [7.3, 2.9, 6.3, 1.8]]}' | python -m json.tool

echo -e "\n🎉 All endpoints working! API PID: $NEW_API_PID"
echo "🛑 To stop: kill $NEW_API_PID"
