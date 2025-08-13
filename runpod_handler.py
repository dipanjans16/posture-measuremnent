"""
RunPod Serverless Handler for Posture Measurement API

This handler adapts the FastAPI-based posture analysis service for RunPod's serverless environment.
It processes dual-image requests (front and side views) and returns posture angle measurements.
"""

import runpod
import json
import base64
import io
import traceback
from typing import Dict, Any

# Import your existing posture analyzer
from measure.posture_analyzer import PostureAnalyzer
from utils.validators import validate_image_content


# Initialize the posture analyzer once when the container starts
print("Initializing PostureAnalyzer...")
posture_analyzer = PostureAnalyzer()
print("PostureAnalyzer initialized successfully!")


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode base64 encoded image to bytes.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        Raw image bytes
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def validate_input(job_input: Dict[str, Any]) -> tuple:
    """
    Validate and extract input data from job request.
    
    Args:
        job_input: The input dictionary from RunPod
        
    Returns:
        Tuple of (front_image_data, side_image_data) as bytes
        
    Raises:
        ValueError: If validation fails
    """
    # Support multiple input formats for flexibility
    front_image_key = None
    side_image_key = None
    
    # Check for the simplified format first
    if "front" in job_input and "side" in job_input:
        front_image_key = "front"
        side_image_key = "side"
    # Check for the detailed format
    elif "front_image" in job_input and "side_image" in job_input:
        front_image_key = "front_image"
        side_image_key = "side_image"
    else:
        available_keys = list(job_input.keys())
        raise ValueError(
            f"Missing required image fields. "
            f"Expected either ('front', 'side') or ('front_image', 'side_image'). "
            f"Available keys: {available_keys}"
        )
    
    # Decode images
    front_image_data = decode_base64_image(job_input[front_image_key])
    side_image_data = decode_base64_image(job_input[side_image_key])
    
    # Validate image content
    validate_image_content(front_image_data)
    validate_image_content(side_image_data)
    
    return front_image_data, side_image_data


def handler(job):
    """
    RunPod serverless handler function.
    
    Expected input formats (both supported):
    
    Simplified format:
    {
        "front": "base64_encoded_image_string",
        "side": "base64_encoded_image_string"
    }
    
    Detailed format:
    {
        "front_image": "base64_encoded_image_string", 
        "side_image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "status": "success" | "error",
        "data": {
            "front_view": {
                "shoulder_tilt": float,
                "pelvic_tilt": float,
                "head_tilt": float,
                "body_tilt": float
            },
            "side_view": {
                "back_tilt": float,
                "neck_tilt": float,
                "body_tilt": float,
                "hip_tilt": float
            }
        },
        "message": str,
        "error": str (only if status is "error")
    }
    """
    try:
        print("Processing new posture measurement request...")
        
        # Get job input
        job_input = job["input"]
        print(f"Received job input with keys: {list(job_input.keys())}")
        
        # Validate and extract input
        front_image_data, side_image_data = validate_input(job_input)
        print("Input validation successful")
        
        # Analyze posture using your existing analyzer
        print("Starting posture analysis...")
        posture_result = posture_analyzer.analyze_posture(
            front_image_data=front_image_data,
            side_image_data=side_image_data
        )
        print("Posture analysis completed successfully")
        
        # Convert Pydantic model to dictionary for JSON serialization
        result_dict = {
            "front_view": {
                "shoulder_tilt": posture_result.front_view.shoulder_tilt,
                "pelvic_tilt": posture_result.front_view.pelvic_tilt,
                "head_tilt": posture_result.front_view.head_tilt,
                "body_tilt": posture_result.front_view.body_tilt
            },
            "side_view": {
                "back_tilt": posture_result.side_view.back_tilt,
                "neck_tilt": posture_result.side_view.neck_tilt,
                "body_tilt": posture_result.side_view.body_tilt,
                "hip_tilt": posture_result.side_view.hip_tilt
            }
        }
        
        return {
            "status": "success",
            "data": result_dict,
            "message": "Posture analysis completed successfully"
        }
        
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "error": f"Input validation failed: {str(e)}",
            "message": "Please check your input format and try again"
        }
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "status": "error", 
            "error": f"Internal server error: {str(e)}",
            "message": "An unexpected error occurred during processing"
        }


# Start the RunPod serverless function
if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
