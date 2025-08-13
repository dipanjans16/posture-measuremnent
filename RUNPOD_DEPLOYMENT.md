# RunPod Deployment Guide for Posture Measurement API

This guide provides comprehensive instructions for deploying your MediaPipe-based posture analysis API on RunPod's serverless platform.

## Overview

Your posture measurement API has been adapted for RunPod serverless deployment with the following components:

- **RunPod Handler** (`runpod_handler.py`): Serverless function that processes base64-encoded images
- **Docker Container** (`Dockerfile`): Optimized container with MediaPipe and OpenCV dependencies
- **Requirements** (`requirements_runpod.txt`): RunPod-specific dependencies including RunPod SDK
- **Testing Script** (`test_runpod_local.py`): Local testing before deployment
- **Build Script** (`build_and_deploy.sh`): Automated deployment script

## Prerequisites

Before deploying, ensure you have:

1. **Docker Desktop** installed and running
2. **RunPod Account** with API access
3. **Container Registry** (Docker Hub, RunPod Registry, or private registry)
4. **Demo Images** in `demo_images/` folder for testing

## Step-by-Step Deployment

### Step 1: Prepare Your Environment

1. **Install Docker** (if not already installed):
   ```bash
   # Download from https://www.docker.com/products/docker-desktop
   ```

2. **Login to your container registry**:
   ```bash
   # For Docker Hub
   docker login
   
   # For RunPod Registry
   docker login registry.runpod.io
   ```

3. **Update registry configuration**:
   Edit `build_and_deploy.sh` and update these variables:
   ```bash
   REGISTRY_URL="your-registry-url"  # e.g., "your-dockerhub-username" or "registry.runpod.io/your-username"
   IMAGE_NAME="posture-measurement-api"
   IMAGE_TAG="latest"
   ```

### Step 2: Test Locally (Recommended)

Before deploying, test the handler locally:

1. **Install dependencies**:
   ```bash
   pip install -r requirements_runpod.txt
   ```

2. **Run local test**:
   ```bash
   python test_runpod_local.py
   ```

   This will:
   - Test the handler with demo images
   - Validate error handling
   - Show expected output format

### Step 3: Build and Deploy

1. **Make the build script executable**:
   ```bash
   chmod +x build_and_deploy.sh
   ```

2. **Run the deployment script**:
   ```bash
   ./build_and_deploy.sh
   ```

   This script will:
   - Build the Docker image with all dependencies
   - Push to your container registry
   - Provide the image URL for RunPod configuration

### Step 4: Configure RunPod Endpoint

1. **Go to RunPod Dashboard**:
   Visit [https://www.runpod.io/console/serverless](https://www.runpod.io/console/serverless)

2. **Create New Endpoint**:
   - Click "New Endpoint"
   - Choose "Custom Image"

3. **Configure the Endpoint**:
   ```yaml
   Image URI: your-registry-url/posture-measurement-api:latest
   
   Environment Variables:
   - PYTHONUNBUFFERED=1
   - PYTHONDONTWRITEBYTECODE=1
   
   Resources:
   - CPU: 2 cores (recommended)
   - Memory: 4GB (minimum for MediaPipe)
   - Storage: 10GB
   
   Scaling:
   - Min Workers: 0 (cost-effective)
   - Max Workers: 10 (adjust based on needs)
   - Idle Timeout: 30 seconds
   
   Timeouts:
   - Request Timeout: 120 seconds
   - Startup Timeout: 300 seconds
   ```

4. **Deploy the Endpoint**:
   - Click "Deploy"
   - Wait for the endpoint to become active

## Usage

### Input Format

Send POST requests to your RunPod endpoint. The handler supports two input formats:

**Simplified format (recommended):**
```json
{
  "front": "base64_encoded_image_string",
  "side": "base64_encoded_image_string"
}
```

**Detailed format (also supported):**
```json
{
  "input": {
    "front_image": "base64_encoded_image_string",
    "side_image": "base64_encoded_image_string"
  }
}
```

### Output Format

The API returns posture angles in this format:

```json
{
  "status": "success",
  "data": {
    "front_view": {
      "shoulder_tilt": 2.34,
      "pelvic_tilt": -1.23,
      "head_tilt": 0.45,
      "body_tilt": 1.67
    },
    "side_view": {
      "back_tilt": 5.67,
      "neck_tilt": 15.23,
      "body_tilt": 3.45,
      "hip_tilt": -2.89
    }
  },
  "message": "Posture analysis completed successfully"
}
```

### Example Usage (Python)

```python
import requests
import base64
import json

# Encode images to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Prepare request
front_image_b64 = encode_image("front_image.jpg")
side_image_b64 = encode_image("side_image.jpg")

# Using simplified format (recommended)
payload = {
    "front": front_image_b64,
    "side": side_image_b64
}

# Or using detailed format:
# payload = {
#     "input": {
#         "front_image": front_image_b64,
#         "side_image": side_image_b64
#     }
# }

# Make request to RunPod endpoint
response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json=payload
)

result = response.json()
print(json.dumps(result, indent=2))
```

## Monitoring and Troubleshooting

### Logs

- View logs in RunPod dashboard under "Logs" tab
- Check container startup logs for initialization issues
- Monitor request processing logs for debugging

### Common Issues

1. **"Could not detect pose landmarks"**:
   - Ensure images clearly show the person
   - Check image quality and lighting
   - Verify person is visible from front/side as required

2. **"Image validation failed"**:
   - Check base64 encoding is correct
   - Ensure images are JPEG/PNG format
   - Verify minimum image size (100x100 pixels)

3. **Container startup timeout**:
   - Increase startup timeout in RunPod settings
   - Check if all dependencies are properly installed
   - Verify Docker image was built successfully

4. **Memory issues**:
   - Increase memory allocation in RunPod settings
   - MediaPipe requires at least 4GB RAM for optimal performance

### Performance Optimization

1. **Cold Start Optimization**:
   - Set minimum workers > 0 for faster response times
   - Consider keeping 1 worker warm for production

2. **Resource Tuning**:
   - Monitor CPU/memory usage
   - Adjust scaling thresholds based on load patterns
   - Use GPU instances if higher throughput needed

## Cost Optimization

- **Idle Timeout**: Set to 30-60 seconds to minimize costs
- **Min Workers**: Keep at 0 for development, consider 1 for production
- **Resource Allocation**: Start with minimum and scale up based on needs

## Security Considerations

1. **API Keys**: Store RunPod API keys securely
2. **Image Data**: Images are processed in memory and not stored
3. **Network**: Use HTTPS for all API communications
4. **Access Control**: Implement authentication in your client applications

## Files Created

The following files have been created for your RunPod deployment:

| File | Purpose |
|------|---------|
| `runpod_handler.py` | Main serverless handler function |
| `Dockerfile` | Container configuration with optimized dependencies |
| `requirements_runpod.txt` | RunPod-specific Python dependencies |
| `build_and_deploy.sh` | Automated build and deployment script |
| `test_runpod_local.py` | Local testing script |
| `runpod_config.json` | Configuration template for RunPod |
| `RUNPOD_DEPLOYMENT.md` | This deployment guide |

## Next Steps

1. **Test the deployed endpoint** with your demo images
2. **Monitor performance** and adjust resources as needed
3. **Implement client applications** using the API
4. **Set up monitoring** and alerting for production use
5. **Consider CI/CD pipeline** for automated deployments

## Support

For issues with:
- **RunPod Platform**: Check RunPod documentation or support
- **MediaPipe/OpenCV**: Refer to respective documentation
- **API Logic**: Review the posture analyzer implementation

## Updates and Maintenance

To update your deployment:

1. Make changes to your code
2. Test locally with `python test_runpod_local.py`
3. Run `./build_and_deploy.sh` to rebuild and push
4. Update the endpoint in RunPod dashboard with new image
5. Test the updated endpoint

Your posture measurement API is now ready for serverless deployment on RunPod! 🚀
