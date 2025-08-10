#!/bin/bash
# debug_and_fix.sh - Debug and fix the API startup issues

echo "🔍 Debugging IRIS API startup..."

# First, let's check if the API file exists and is correct
echo "📁 Checking if API file exists..."
if [ -f "src/api.py" ]; then
    echo "✅ src/api.py exists"
    echo "📏 File size: $(wc -l < src/api.py) lines"
else
    echo "❌ src/api.py not found!"
    exit 1
fi

# Check if required directories exist
echo "📂 Checking required model files..."
if [ -f "models/iris_model.pkl" ]; then
    echo "✅ Model file exists"
else
    echo "❌ Model file not found at models/iris_model.pkl"
    echo "🔧 Available files in models/:"
    ls -la models/ || echo "Models directory doesn't exist"
fi

if [ -f "data/processed/scaler.pkl" ]; then
    echo "✅ Scaler file exists"
else
    echo "⚠️  Scaler file not found at data/processed/scaler.pkl"
    echo "🔧 Available files in data/processed/:"
    ls -la data/processed/ || echo "Processed data directory doesn't exist"
fi

# Kill any existing Python processes on port 8080
echo "🧹 Cleaning up any existing processes on port 8080..."
pkill -f "api.py" 2>/dev/null || true
pkill -f "python.*8080" 2>/dev/null || true

# Wait a moment
sleep 2

# Check if port 8080 is free
echo "🔍 Checking if port 8080 is available..."
if netstat -tuln | grep :8080 > /dev/null; then
    echo "⚠️  Port 8080 is still in use"
    echo "🔧 Processes using port 8080:"
    lsof -i :8080 || netstat -tuln | grep :8080
else
    echo "✅ Port 8080 is available"
fi

# Set correct environment variables
echo "🔧 Setting environment variables..."
export GCP_PROJECT_ID="elaborate-chess-461609-j9"
export GKE_CLUSTER_NAME="iris-mlops-cluster"
export GCP_ZONE="us-central1-a"

echo "✅ Environment variables set:"
echo "   PROJECT_ID: $GCP_PROJECT_ID"
echo "   CLUSTER: $GKE_CLUSTER_NAME"
echo "   ZONE: $GCP_ZONE"

# Try to start the API with verbose output
echo "🚀 Starting the API with debug output..."
cd ~/iris-ml-pipline

# Start the API in background and capture the PID
echo "📝 Starting API server..."
python src/api.py > api.log 2>&1 &
API_PID=$!

echo "🔢 API started with PID: $API_PID"

# Wait a few seconds for the server to start
echo "⏳ Waiting for API to start..."
sleep 5

# Check if the process is still running
if ps -p $API_PID > /dev/null; then
    echo "✅ API process is running"
else
    echo "❌ API process died. Checking logs..."
    cat api.log
    exit 1
fi

# Check what's actually running on port 8080
echo "🔍 Checking what's listening on port 8080..."
netstat -tuln | grep :8080 || echo "Nothing listening on 8080"

# Try to connect to the API
echo "🧪 Testing API endpoints..."

# Test health endpoint
echo "📋 Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" http://127.0.0.1:8080/health 2>/dev/null || echo "CONNECTION_FAILED")

if [[ "$HEALTH_RESPONSE" == *"200"* ]]; then
    echo "✅ Health check passed"
    echo "Response: $HEALTH_RESPONSE"
else
    echo "❌ Health check failed"
    echo "Response: $HEALTH_RESPONSE"
    
    # Let's check the API logs
    echo "📄 API logs:"
    cat api.log
    
    # Try using different localhost addresses
    echo "🔄 Trying localhost alternatives..."
    curl -s http://localhost:8080/health || echo "localhost failed"
    curl -s http://0.0.0.0:8080/health || echo "0.0.0.0 failed"
fi

# Test prediction endpoint if health check passed
if [[ "$HEALTH_RESPONSE" == *"200"* ]]; then
    echo "🎯 Testing prediction endpoint..."
    PRED_RESPONSE=$(curl -s -X POST http://127.0.0.1:8080/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [5.1, 3.5, 1.4, 0.2]}' 2>/dev/null || echo "PREDICTION_FAILED")
    
    echo "🎯 Prediction response: $PRED_RESPONSE"
fi

# Show API logs
echo "📄 Current API logs:"
cat api.log

# Keep the API running for manual testing
echo "🎮 API is running for manual testing..."
echo "📋 Test commands:"
echo "   curl http://127.0.0.1:8080/health"
echo "   curl -X POST http://127.0.0.1:8080/predict -H 'Content-Type: application/json' -d '{\"features\": [5.1, 3.5, 1.4, 0.2]}'"
echo ""
echo "🛑 To stop the API, run: kill $API_PID"
echo "📄 To view logs, run: tail -f api.log"
