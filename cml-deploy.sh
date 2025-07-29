#!/bin/bash
# cml-deploy.sh - Continuous Deployment script using CML for IRIS API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
CLUSTER_NAME=${GKE_CLUSTER_NAME:-"iris-mlops-cluster"}
ZONE=${GCP_ZONE:-"us-central1-a"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/iris-api"
IMAGE_TAG=${GITHUB_SHA:-$(date +%Y%m%d%H%M%S)}
NAMESPACE="iris-api"

echo -e "${GREEN}🚀 Starting IRIS API Continuous Deployment with CML${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Project ID: ${PROJECT_ID}"
echo -e "  Cluster: ${CLUSTER_NAME}"
echo -e "  Zone: ${ZONE}"
echo -e "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install required tools if not present
echo -e "${YELLOW}🔧 Checking and installing required tools...${NC}"

if ! command_exists gcloud; then
    echo "Installing Google Cloud SDK..."
    curl https://sdk.cloud.google.com | bash
    exec -l $SHELL
fi

if ! command_exists kubectl; then
    echo "Installing kubectl..."
    gcloud components install kubectl
fi

if ! command_exists docker; then
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo usermod -aG docker $USER
fi

# Authenticate with GCP (if service account key is provided)
if [ ! -z "$GCP_SERVICE_ACCOUNT_KEY" ]; then
    echo -e "${YELLOW}🔐 Authenticating with GCP...${NC}"
    echo $GCP_SERVICE_ACCOUNT_KEY | base64 -d > gcp-key.json
    gcloud auth activate-service-account --key-file gcp-key.json
fi

gcloud config set project $PROJECT_ID

# Configure Docker for GCR
echo -e "${YELLOW}🐳 Configuring Docker for Google Container Registry...${NC}"
gcloud auth configure-docker

# Run DVC pipeline to ensure latest model
echo -e "${YELLOW}📊 Running DVC pipeline to get latest model...${NC}"
if [ -f "dvc.yaml" ]; then
    dvc repro
    echo -e "${GREEN}✅ DVC pipeline completed${NC}"
else
    echo -e "${YELLOW}⚠️  No dvc.yaml found, skipping DVC repro${NC}"
fi

# Generate CML report header
echo -e "${YELLOW}📋 Generating CML report...${NC}"
cat > cml_report.md << EOF
# 🌸 IRIS API Deployment Report

## 📊 Model Performance
EOF

# Add model metrics to report if available
if [ -f "models/metrics.json" ]; then
    python3 << EOF
import json
import os

try:
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    report = f"""
### Current Model Metrics
- **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}
- **Precision**: {metrics.get('precision', 'N/A'):.4f}
- **Recall**: {metrics.get('recall', 'N/A'):.4f}
- **F1-Score**: {metrics.get('f1_score', 'N/A'):.4f}

### Model Information
- **Model Type**: Iris Classification
- **Features**: sepal_length, sepal_width, petal_length, petal_width
- **Classes**: setosa, versicolor, virginica
"""
    
    with open('cml_report.md', 'a') as f:
        f.write(report)
        
except Exception as e:
    with open('cml_report.md', 'a') as f:
        f.write(f"\n⚠️ Could not load metrics: {str(e)}\n")
EOF
else
    echo "⚠️ No metrics file found" >> cml_report.md
fi

# Add deployment info to report
cat >> cml_report.md << EOF

## 🚀 Deployment Information
- **Image**: \`${IMAGE_NAME}:${IMAGE_TAG}\`
- **Cluster**: ${CLUSTER_NAME}
- **Zone**: ${ZONE}
- **Namespace**: ${NAMESPACE}
- **Replicas**: 3
- **Deployment Time**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

EOF

# Build Docker image
echo -e "${YELLOW}🏗️  Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest

echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Push to Google Container Registry
echo -e "${YELLOW}📤 Pushing image to Google Container Registry...${NC}"
docker push ${IMAGE_NAME}:${IMAGE_TAG}
docker push ${IMAGE_NAME}:latest

echo -e "${GREEN}✅ Image pushed to GCR${NC}"

# Create GKE cluster if it doesn't exist
echo -e "${YELLOW}🏭 Checking/Creating GKE cluster...${NC}"
if ! gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE >/dev/null 2>&1; then
    echo "Creating GKE cluster..."
    gcloud container clusters create $CLUSTER_NAME \
        --zone=$ZONE \
        --num-nodes=3 \
        --enable-autorepair \
        --enable-autoupgrade \
        --machine-type=e2-medium \
        --disk-size=20GB \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=5
    echo -e "${GREEN}✅ GKE cluster created${NC}"
else
    echo -e "${GREEN}✅ GKE cluster already exists${NC}"
fi

# Get GKE cluster credentials
echo -e "${YELLOW}🔑 Getting GKE cluster credentials...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Create k8s directory if it doesn't exist
mkdir -p k8s

# Update Kubernetes manifests with correct project ID and image tag
echo -e "${YELLOW}📝 Updating Kubernetes manifests...${NC}"
sed -i "s/YOUR_PROJECT_ID/${PROJECT_ID}/g" k8s/*.yaml
sed -i "s/:latest/:${IMAGE_TAG}/g" k8s/deployment.yaml

# Deploy to Kubernetes
echo -e "${YELLOW}☸️  Deploying to Kubernetes...${NC}"

# Apply namespace first
kubectl apply -f k8s/namespace.yaml

# Apply all other manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/configmap.yaml

echo -e "${GREEN}✅ Kubernetes manifests applied${NC}"

# Wait for deployment to be ready
echo -e "${YELLOW}⏳ Waiting for deployment to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/iris-api -n $NAMESPACE

# Get deployment status
DEPLOYMENT_STATUS=$(kubectl get deployment iris-api -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Available")].status}')

if [ "$DEPLOYMENT_STATUS" = "True" ]; then
    echo -e "${GREEN}✅ Deployment is ready${NC}"
    DEPLOYMENT_SUCCESS="✅ SUCCESS"
else
    echo -e "${RED}❌ Deployment failed${NC}"
    DEPLOYMENT_SUCCESS="❌ FAILED"
fi

# Get service external IP
echo -e "${YELLOW}🌐 Getting service external IP...${NC}"
EXTERNAL_IP=""
TIMEOUT=300
ELAPSED=0

while [ -z "$EXTERNAL_IP" ] && [ $ELAPSED -lt $TIMEOUT ]; do
    echo "Waiting for external IP... (${ELAPSED}s/${TIMEOUT}s)"
    EXTERNAL_IP=$(kubectl get svc iris-api-service -n $NAMESPACE --template="{{range .status.loadBalancer.ingress}}{{.ip}}{{end}}" 2>/dev/null || echo "")
    if [ -z "$EXTERNAL_IP" ]; then
        sleep 10
        ELAPSED=$((ELAPSED + 10))
    fi
done

if [ -z "$EXTERNAL_IP" ]; then
    echo -e "${RED}❌ Failed to get external IP${NC}"
    EXTERNAL_IP="Pending..."
    API_STATUS="❌ No External IP"
else
    echo -e "${GREEN}✅ External IP obtained: ${EXTERNAL_IP}${NC}"
    API_STATUS="✅ Available"
fi

# Update CML report with deployment results
cat >> cml_report.md << EOF
## 📋 Deployment Status
- **Status**: ${DEPLOYMENT_SUCCESS}
- **External IP**: \`${EXTERNAL_IP}\`
- **API Status**: ${API_STATUS}

### 🔗 API Endpoints
- **Health Check**: http://${EXTERNAL_IP}/health
- **Prediction**: http://${EXTERNAL_IP}/predict
- **Batch Prediction**: http://${EXTERNAL_IP}/batch_predict
- **Model Info**: http://${EXTERNAL_IP}/model_info

EOF

# Test the deployed API if external IP is available
if [ "$EXTERNAL_IP" != "Pending..." ]; then
    echo -e "${YELLOW}🧪 Testing deployed API...${NC}"
    sleep 30  # Give time for pods to be ready
    
    # Health check
    echo "Testing health endpoint..."
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${EXTERNAL_IP}/health --connect-timeout 10 --max-time 30 || echo "000")
    
    if [ "$HEALTH_STATUS" = "200" ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
        HEALTH_RESULT="✅ PASSED (HTTP $HEALTH_STATUS)"
    else
        echo -e "${RED}❌ Health check failed (HTTP $HEALTH_STATUS)${NC}"
        HEALTH_RESULT="❌ FAILED (HTTP $HEALTH_STATUS)"
    fi
    
    # Prediction test
    echo "Testing prediction endpoint..."
    PREDICTION_RESPONSE=$(curl -s -X POST http://${EXTERNAL_IP}/predict \
        -H "Content-Type: application/json" \
        -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
        --connect-timeout 10 --max-time 30 || echo "ERROR")
    
    if [[ "$PREDICTION_RESPONSE" == *"success"* ]]; then
        echo -e "${GREEN}✅ Prediction test passed${NC}"
        PREDICTION_RESULT="✅ PASSED"
    else
        echo -e "${RED}❌ Prediction test failed${NC}"
        PREDICTION_RESULT="❌ FAILED"
    fi
    
    # Add test results to report
    cat >> cml_report.md << EOF

## 🧪 API Testing Results
- **Health Check**: ${HEALTH_RESULT}
- **Prediction Test**: ${PREDICTION_RESULT}

### 📝 Sample API Usage
\`\`\`bash
# Health check
curl http://${EXTERNAL_IP}/health

# Single prediction
curl -X POST http://${EXTERNAL_IP}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Batch prediction
curl -X POST http://${EXTERNAL_IP}/batch_predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}'
\`\`\`

EOF

    if [[ "$PREDICTION_RESPONSE" == *"success"* ]]; then
        echo "### 🎯 Sample Prediction Response" >> cml_report.md
        echo '```json' >> cml_report.md
        echo "$PREDICTION_RESPONSE" | python3 -m json.tool >> cml_report.md 2>/dev/null || echo "$PREDICTION_RESPONSE" >> cml_report.md
        echo '```' >> cml_report.md
    fi
else
    cat >> cml_report.md << EOF

## ⚠️ API Testing
Could not test API endpoints as external IP is not available yet.
Please check the service status with:
\`\`\`bash
kubectl get svc iris-api-service -n ${NAMESPACE}
\`\`\`

EOF
fi

# Add resource information
cat >> cml_report.md << EOF

## 📊 Kubernetes Resources
\`\`\`bash
# Check deployment status
kubectl get deployment iris-api -n ${NAMESPACE}

# Check pods
kubectl get pods -n ${NAMESPACE} -l app=iris-api

# Check service
kubectl get svc iris-api-service -n ${NAMESPACE}

# Check HPA status
kubectl get hpa iris-api-hpa -n ${NAMESPACE}
\`\`\`

## 🏁 Deployment Summary
- **Docker Image**: Built and pushed to GCR
- **Kubernetes Deployment**: Applied with 3 replicas
- **Load Balancer**: Service exposed externally
- **Auto Scaling**: HPA configured (2-10 replicas)
- **Health Checks**: Configured for liveness and readiness

---
*Deployment completed at $(date -u +"%Y-%m-%d %H:%M:%S UTC")*
EOF

# Send CML report (if cml is available)
if command_exists cml; then
    echo -e "${YELLOW}📤 Sending CML report...${NC}"
    cml comment create cml_report.md
else
    echo -e "${YELLOW}📄 CML report generated: cml_report.md${NC}"
fi

# Summary
echo -e "${GREEN}🎉 CML Deployment Pipeline Completed!${NC}"
echo -e "${BLUE}Summary:${NC}"
echo -e "  ✅ Model pipeline executed"
echo -e "  ✅ Docker image built and pushed"
echo -e "  ✅ Kubernetes deployment applied"
echo -e "  ✅ Service exposed with LoadBalancer"
echo -e "  📊 CML report generated"

if [ "$EXTERNAL_IP" != "Pending..." ]; then
    echo -e "${GREEN}🌐 API available at: http://${EXTERNAL_IP}${NC}"
else
    echo -e "${YELLOW}⏳ External IP pending, check with: kubectl get svc iris-api-service -n ${NAMESPACE}${NC}"
fi

echo -e "${BLUE}📄 Full report available in: cml_report.md${NC}"