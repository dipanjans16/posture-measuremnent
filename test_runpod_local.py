#!/usr/bin/env python3
"""
Local testing script for RunPod handler
Tests the handler function locally before deployment
"""

import base64
import json
from pathlib import Path

# Import the handler
from runpod_handler import handler


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_posture_measurement():
    """Test the posture measurement endpoint locally."""
    
    print("🧪 Testing RunPod handler locally...")
    
    # Paths to demo images
    front_image_path = "demo_images/front_image.jpg"
    side_image_path = "demo_images/side_image.jpg"
    
    # Check if demo images exist
    if not Path(front_image_path).exists():
        print(f"❌ Demo image not found: {front_image_path}")
        print("Please ensure demo images are available in the demo_images folder")
        return
        
    if not Path(side_image_path).exists():
        print(f"❌ Demo image not found: {side_image_path}")
        print("Please ensure demo images are available in the demo_images folder")
        return
    
    try:
        # Encode images to base64
        print("📷 Encoding images...")
        front_image_b64 = encode_image_to_base64(front_image_path)
        side_image_b64 = encode_image_to_base64(side_image_path)
        
        # Create test job input (using simplified format)
        test_job = {
            "input": {
                "front": front_image_b64,
                "side": side_image_b64
            }
        }
        
        print("🔄 Processing request...")
        
        # Call the handler
        result = handler(test_job)
        
        # Print results
        print("\n📊 Results:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        print("=" * 50)
        
        if result["status"] == "success":
            print("✅ Test completed successfully!")
            
            # Print angles in a readable format
            data = result["data"]
            print("\n📐 Posture Angles:")
            print(f"Front View:")
            print(f"  - Shoulder Tilt: {data['front_view']['shoulder_tilt']:.2f}°")
            print(f"  - Pelvic Tilt: {data['front_view']['pelvic_tilt']:.2f}°")
            print(f"  - Head Tilt: {data['front_view']['head_tilt']:.2f}°")
            print(f"  - Body Tilt: {data['front_view']['body_tilt']:.2f}°")
            
            print(f"Side View:")
            print(f"  - Back Tilt: {data['side_view']['back_tilt']:.2f}°")
            print(f"  - Neck Tilt: {data['side_view']['neck_tilt']:.2f}°")
            print(f"  - Body Tilt: {data['side_view']['body_tilt']:.2f}°")
            print(f"  - Hip Tilt: {data['side_view']['hip_tilt']:.2f}°")
            
        else:
            print("❌ Test failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling with invalid input."""
    
    print("\n🧪 Testing error handling...")
    
    # Test with missing front image
    test_job_missing_front = {
        "input": {
            "side": "invalid_base64"
        }
    }
    
    result = handler(test_job_missing_front)
    print(f"Missing front_image result: {result['status']}")
    assert result["status"] == "error"
    
    # Test with invalid base64
    test_job_invalid_b64 = {
        "input": {
            "front": "invalid_base64",
            "side": "invalid_base64"
        }
    }
    
    result = handler(test_job_invalid_b64)
    print(f"Invalid base64 result: {result['status']}")
    assert result["status"] == "error"
    
    print("✅ Error handling tests passed!")


if __name__ == "__main__":
    print("🏃‍♂️ Running local tests for RunPod handler...")
    print()
    
    # Test normal operation
    test_posture_measurement()
    
    # Test error handling
    test_error_handling()
    
    print("\n🎉 All tests completed!")
