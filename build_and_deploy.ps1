# RunPod Deployment Script for Posture Measurement API (PowerShell)
# This script builds and pushes the Docker image to a container registry

# Configuration - UPDATE THESE VALUES
$REGISTRY_URL = "dipanjan16"  # e.g., "your-dockerhub-username" or "registry.runpod.io/your-username"
$IMAGE_NAME = "posture-measurement-api"
$IMAGE_TAG = "latest"
$FULL_IMAGE_NAME = "${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"

Write-Host "🚀 Starting RunPod deployment process..." -ForegroundColor Green
Write-Host "Building image: $FULL_IMAGE_NAME" -ForegroundColor Yellow

# Step 1: Build the Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Blue
docker build -t $FULL_IMAGE_NAME .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Push to registry
Write-Host "📤 Pushing image to registry..." -ForegroundColor Blue
docker push $FULL_IMAGE_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Image pushed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 Deployment preparation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Go to RunPod dashboard (https://www.runpod.io/console/serverless)"
    Write-Host "2. Create a new serverless endpoint"
    Write-Host "3. Use this image: $FULL_IMAGE_NAME"
    Write-Host "4. Set environment variables if needed"
    Write-Host "5. Configure your endpoint settings"
    Write-Host ""
    Write-Host "📋 Image details:" -ForegroundColor Cyan
    Write-Host "   Registry: $REGISTRY_URL"
    Write-Host "   Image: $IMAGE_NAME"
    Write-Host "   Tag: $IMAGE_TAG"
    Write-Host "   Full name: $FULL_IMAGE_NAME"
} else {
    Write-Host "❌ Failed to push image to registry!" -ForegroundColor Red
    exit 1
}

