#!/usr/bin/env python3
"""
Test script for core handler functionality without RunPod dependencies
"""

import base64
import json
from pathlib import Path

# Import only the core functions we need
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the components we need to test
from measure.posture_analyzer import PostureAnalyzer
from utils.validators import validate_image_content


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def decode_base64_image(base64_string: str) -> bytes:
    """Decode base64 encoded image to bytes."""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        return base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def validate_input(job_input: dict) -> tuple:
    """Validate and extract input data from job request."""
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


def test_core_functionality():
    """Test the core posture analysis functionality."""
    
    print("🧪 Testing core handler functionality...")
    
    # Initialize analyzer
    print("Initializing PostureAnalyzer...")
    posture_analyzer = PostureAnalyzer()
    print("PostureAnalyzer initialized successfully!")
    
    # Paths to demo images
    front_image_path = "demo_images/front_image.jpg"
    side_image_path = "demo_images/side_image.jpg"
    
    # Check if demo images exist
    if not Path(front_image_path).exists():
        print(f"❌ Demo image not found: {front_image_path}")
        return False
        
    if not Path(side_image_path).exists():
        print(f"❌ Demo image not found: {side_image_path}")
        return False
    
    try:
        # Encode images to base64
        print("📷 Encoding images...")
        front_image_b64 = encode_image_to_base64(front_image_path)
        side_image_b64 = encode_image_to_base64(side_image_path)
        
        # Test simplified format
        print("🔄 Testing simplified format...")
        test_input_simple = {
            "front": front_image_b64,
            "side": side_image_b64
        }
        
        front_data, side_data = validate_input(test_input_simple)
        posture_result = posture_analyzer.analyze_posture(
            front_image_data=front_data,
            side_image_data=side_data
        )
        
        # Convert to dict for display
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
        
        print("✅ Simplified format test passed!")
        print("\n📊 Results:")
        print("=" * 50)
        print(json.dumps(result_dict, indent=2))
        print("=" * 50)
        
        # Test detailed format
        print("\n🔄 Testing detailed format...")
        test_input_detailed = {
            "front_image": front_image_b64,
            "side_image": side_image_b64
        }
        
        front_data, side_data = validate_input(test_input_detailed)
        posture_result2 = posture_analyzer.analyze_posture(
            front_image_data=front_data,
            side_image_data=side_data
        )
        
        print("✅ Detailed format test passed!")
        
        # Test error handling
        print("\n🧪 Testing error handling...")
        try:
            validate_input({"invalid": "data"})
            print("❌ Error handling failed - should have thrown an error")
            return False
        except ValueError as e:
            print(f"✅ Error handling works: {str(e)}")
        
        print("\n🎉 All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_core_functionality()
    if success:
        print("\n✅ Core functionality is working correctly!")
        print("You can now test with the RunPod handler using:")
        print("python runpod_handler.py --rp_serve_api")
    else:
        print("\n❌ Tests failed. Please check the errors above.")
