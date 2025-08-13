"""
API routes for posture measurement endpoint.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import time

from models.schemas import PostureAngles
from utils.validators import validate_image_files
from measure.posture_analyzer import PostureAnalyzer

router = APIRouter()
posture_analyzer = PostureAnalyzer()

@router.get("/")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Posture measurement API is running"}


@router.post("/measure_posture")
async def measure_posture_angles(
    front_image: UploadFile = File(
        ..., description="Front view image of the person (REQUIRED)"
    ),
    side_image: UploadFile = File(
        ..., description="Side view image of the person (REQUIRED)"
    ),
):
    """
    Analyze posture angles using TWO images (front and side view)

    This endpoint uses MediaPipe pose detection with maximum model complexity (2) 
    to extract pose landmarks and calculate posture angles.

    Parameters:
    - front_image: Front facing image file showing the person from the front
    - side_image: Side facing image file showing the person from the side

    Returns:
    - Posture angles for both front and side views as specified:
      - Front view: shoulder_tilt, pelvic_tilt, head_tilt, body_tilt
      - Side view: pelvic_tilt, head_tilt, body_tilt
    """
    # Record start time
    start_time = time.time()
    
    # Validate image files
    validate_image_files(front_image, side_image)

    # Read image data
    front_image_data = await front_image.read()
    side_image_data = await side_image.read()

    # Analyze posture using MediaPipe
    posture_result = posture_analyzer.analyze_posture(
        front_image_data=front_image_data,
        side_image_data=side_image_data
    )

    # Calculate and print runtime
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Posture measurement runtime: {runtime:.4f} seconds")

    return posture_result
