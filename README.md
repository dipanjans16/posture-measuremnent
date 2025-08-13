# Posture Measurement API

AI-powered posture analysis service using MediaPipe pose detection from dual-image analysis.

## Setup

### Prerequisites
- Python 3.8 or higher
- No CUDA required (uses CPU-based MediaPipe)

### Installation
Please use PowerShell for installation and execution.

#### Create venv
```powershell
py -3.10 -m venv venv
```

If you haven't installed Python 3.10, please install it first.
You can check the installed Python versions by running `py --list`.

#### Activate venv
```powershell
.\venv\Scripts\activate
```

When successful, `(venv)` will be displayed at the beginning of the input line. This indicates that the virtual environment is now active.

#### Install dependencies
Execute this after activating the virtual environment.

```powershell
pip install -r requirements_posture.txt
```

**Note**: This simplified API only requires basic dependencies for posture analysis. No heavy ML models or CUDA setup needed.

## Usage

### Start API Server
Execute this after activating the virtual environment (see [Activate venv](#activate-venv)).

```powershell
python main.py
```
Server runs on `http://localhost:8000`

### Test the API

#### Using Swagger UI
Visit `http://localhost:8000/docs` for interactive API testing.
Use the front_image.jpg and side_image.jpg in demo_images folder for testing.

### API Endpoints

- **GET /**: Health check endpoint
  - Returns API status

- **POST /measure_posture**: Analyzes posture angles from front and side images
  - `front_image` (required): Front view image (JPEG/PNG)
  - `side_image` (required): Side view image (JPEG/PNG)

### Important Side Pose Instructions
When capturing the side pose image, the person should turn to their right from the front pose position. This means the left side of their body should be facing the camera. The pose detection algorithm is optimized for this specific orientation where the left side of the person is visible to the camera.

### Response
The `/measure_posture` endpoint returns posture angles for both views:

**Front view angles:**
- `shoulder_tilt`: Angle between shoulders relative to horizontal
- `pelvic_tilt`: Angle between hips relative to horizontal  
- `head_tilt`: Angle between ears relative to horizontal
- `body_tilt`: Angle from nose to hip center relative to vertical

**Side view angles:**
- `back_tilt`: Back tilt angle between shoulder and hip relative to vertical
- `neck_tilt`: Neck tilt angle between ear and shoulder relative to vertical
- `body_tilt`: Overall body tilt angle from ear to hip relative to vertical
- `hip_tilt`: Hip tilt angle between hip and knee relative to vertical

All angles are returned in degrees.

## What's Different

This is a simplified version of the original body measurement API that focuses **only** on posture analysis:

### ✅ What's Included:
- MediaPipe-based pose detection
- Posture angle calculations for front and side views
- Lightweight FastAPI application
- Minimal dependencies (no CUDA, PyTorch, or SMPL-X models)

### ❌ What's Removed:
- Body measurement calculations (circumferences, lengths)
- SMPL-X model processing  
- CUDA/GPU requirements
- Heavy ML model dependencies
- Body fat percentage calculations

### 🚀 Benefits:
- **Faster startup**: No model loading delays
- **Lower resource usage**: CPU-only processing
- **Easier setup**: No complex model downloads
- **Focused functionality**: Pure posture analysis

## Configuration
Server settings can be modified in `app_config/settings.py`:
- Default host: `0.0.0.0`
- Default port: `8000`
- Supported image formats: JPEG, PNG, JPG
