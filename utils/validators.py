"""
Input validation utilities for posture measurement API.
"""

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from app_config.settings import APIConfig


def validate_image_files(*image_files: UploadFile):
    """
    Validates that uploaded files are valid images in allowed formats.

    Args:
        *image_files: One or more image files to validate. Must be JPEG or PNG
                     format and contain photos for posture analysis.

    Raises:
        HTTPException: If any file is missing or has an invalid format.
                      Returns 400 status with descriptive error message.
    """
    for i, image_file in enumerate(image_files):
        if not image_file:
            raise HTTPException(status_code=400, detail=f"Image file {i+1} is required")

        if image_file.content_type not in APIConfig.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image {i+1} must be JPEG or PNG. "
                    f"Got: {image_file.content_type}"
                ),
            )


def validate_input_parameters(
    height: float, weight: float, sex: str, supported_sexes: list[str] = None
):
    """
    Validates that body measurement input parameters are within valid ranges.

    Args:
        height: Person's height in centimeters. Must be between MIN_HEIGHT and
               MAX_HEIGHT as defined in APIConfig.
        weight: Person's weight in kilograms. Must be between MIN_WEIGHT and
               MAX_WEIGHT as defined in APIConfig.
        sex: Person's biological sex.
        supported_sexes: A list of supported sex strings (e.g. ["male", "female", "neutral"])
                         If None, defaults to ["male", "female"].

    Raises:
        HTTPException: If any parameter is outside valid range. Returns 400
                      status with descriptive error message indicating valid
                      ranges.
    """
    if not (APIConfig.MIN_HEIGHT <= height <= APIConfig.MAX_HEIGHT):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Height must be between {APIConfig.MIN_HEIGHT} "
                f"and {APIConfig.MAX_HEIGHT} cm"
            ),
        )

    if not (APIConfig.MIN_WEIGHT <= weight <= APIConfig.MAX_WEIGHT):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Weight must be between {APIConfig.MIN_WEIGHT} "
                f"and {APIConfig.MAX_WEIGHT} kg"
            ),
        )

    # Default to male/female if no specific list is provided for backward compatibility or simple cases
    if supported_sexes is None:
        supported_sexes = ["male", "female"]

    sex_lower = sex.lower()
    if sex_lower not in [s.lower() for s in supported_sexes]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sex value '{sex}'. Supported values are: {', '.join(supported_sexes)}",
        )


def validate_image_content(image_data: bytes):
    """
    Validates that the image data can be properly decoded and is a valid image.
    
    Args:
        image_data: Raw image bytes to validate
        
    Raises:
        ValueError: If the image data is invalid or cannot be decoded
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Try to decode the image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data - could not decode image")
            
        # Check if image has valid dimensions
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a color image with 3 channels")
            
        # Check minimum image size (more lenient for posture analysis)
        if image.shape[0] < 50 or image.shape[1] < 50:
            raise ValueError("Image must be at least 50x50 pixels")
            
    except Exception as e:
        raise ValueError(f"Image validation failed: {str(e)}")
