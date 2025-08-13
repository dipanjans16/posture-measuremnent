# RunPod Serverless Dockerfile for Posture Measurement API
# Optimized for MediaPipe and OpenCV dependencies

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install essential system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements_runpod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_runpod.txt

# Copy the application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p api app_config core measure models services utils

# Set Python path to include the current directory
ENV PYTHONPATH=/app

# Expose port (for debugging, not used in serverless)
EXPOSE 8000

# Run the RunPod handler
CMD ["python", "runpod_handler.py"]
