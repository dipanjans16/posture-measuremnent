#!/bin/bash

# RunPod Deployment Script for Posture Measurement API
# This script builds and pushes the Docker image to a container registry

set -e  # Exit on any error

# Configuration - UPDATE THESE VALUES
REGISTRY_URL="dipanjan16"  # e.g., "registry.runpod.io/your-username"
IMAGE_NAME="posture-measurement-api"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🚀 Starting RunPod deployment process..."
echo "Building image: ${FULL_IMAGE_NAME}"

# Step 1: Build the Docker image
echo "📦 Building Docker image..."
docker build -t ${FULL_IMAGE_NAME} .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Docker build failed!"
    exit 1
fi

# Step 2: Push to registry
echo "📤 Pushing image to registry..."
docker push ${FULL_IMAGE_NAME}

if [ $? -eq 0 ]; then
    echo "✅ Image pushed successfully!"
    echo ""
    echo "🎉 Deployment preparation complete!"
    echo ""
    echo "Next steps:"
    echo "1. Go to RunPod dashboard (https://www.runpod.io/console/serverless)"
    echo "2. Create a new serverless endpoint"
    echo "3. Use this image: ${FULL_IMAGE_NAME}"
    echo "4. Set environment variables if needed"
    echo "5. Configure your endpoint settings"
    echo ""
    echo "📋 Image details:"
    echo "   Registry: ${REGISTRY_URL}"
    echo "   Image: ${IMAGE_NAME}"
    echo "   Tag: ${IMAGE_TAG}"
    echo "   Full name: ${FULL_IMAGE_NAME}"
else
    echo "❌ Failed to push image to registry!"
    exit 1
fi
