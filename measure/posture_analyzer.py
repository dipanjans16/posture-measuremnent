"""
Posture analysis service using MediaPipe pose detection.
"""

import math
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional
import io

from models.schemas import PostureAngles, FrontViewPosture, SideViewPosture


class PostureAnalyzer:
    """MediaPipe-based posture analyzer."""
    
    def __init__(self):
        """Initialize MediaPipe pose detection with maximum complexity."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Maximum complexity as requested
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # MediaPipe pose landmark indices
        self.KEYPOINT_INDICES = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }

    def calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float], 
                       reference_line: str = 'horizontal') -> float:
        """Calculate angle between two points and a reference line.
        
        Args:
            point1: First point [x, y]
            point2: Second point [x, y]  
            reference_line: 'horizontal' or 'vertical'
            
        Returns:
            Angle in degrees
        """
        if reference_line == 'horizontal':
            # Calculate angle with horizontal line
            dx = point1[0] - point2[0]
            dy = point2[1] - point1[1]
            angle_rad = math.atan2(dy, dx)
        else:  # vertical
            # Calculate angle with vertical line
            dx = point1[0] - point2[0]
            dy = point2[1] - point1[1]
            angle_rad = math.atan2(dx, dy)
        
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def extract_keypoints_from_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Extract pose landmarks from image using MediaPipe.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Array of keypoints [N, 2] where N is number of keypoints, or None if no pose detected
        """
        # Convert bytes to image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
            
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get pose landmarks
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
            
        # Extract landmark coordinates
        landmarks = results.pose_landmarks.landmark
        height, width = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        keypoints = []
        for landmark in landmarks:
            x = landmark.x * width
            y = landmark.y * height
            keypoints.append([x, y])
            
        return np.array(keypoints)

    def calculate_posture_angles(self, keypoints: np.ndarray, view: str = "front") -> Dict[str, float]:
        """Calculate posture angles based on keypoints and view.
        
        Args:
            keypoints: Array of keypoints [N, 2] where N is number of keypoints
            view: "front" or "side"
            
        Returns:
            Dictionary containing posture angles
        """
        posture_angles = {}
        
        if view == "front":
            # Front view calculations
            
            # Shoulder tilt: angle between left_shoulder and right_shoulder with horizontal
            left_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINT_INDICES['right_shoulder']]
            posture_angles['shoulder_tilt'] = self.calculate_angle(left_shoulder, right_shoulder, 'horizontal')
            
            # Pelvic tilt: angle between left_hip and right_hip with horizontal
            left_hip = keypoints[self.KEYPOINT_INDICES['left_hip']]
            right_hip = keypoints[self.KEYPOINT_INDICES['right_hip']]
            posture_angles['pelvic_tilt'] = self.calculate_angle(left_hip, right_hip, 'horizontal')
            
            # Head tilt: angle between left_ear and right_ear with horizontal
            left_ear = keypoints[self.KEYPOINT_INDICES['left_ear']]
            right_ear = keypoints[self.KEYPOINT_INDICES['right_ear']]
            posture_angles['head_tilt'] = self.calculate_angle(left_ear, right_ear, 'horizontal')
            
            # Body tilt: angle between midpoint of hips and midpoint of shoulders with vertical
            hip_midpoint = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
            nose = keypoints[self.KEYPOINT_INDICES['nose']]
            posture_angles['body_tilt'] = self.calculate_angle(nose, hip_midpoint, 'vertical')
            
        elif view == "side":
            # Side view calculations (using left side keypoints for consistency)
            
            # Pelvic tilt: angle between left_hip and left_shoulder with vertical
            left_shoulder = keypoints[self.KEYPOINT_INDICES['left_shoulder']]
            left_hip = keypoints[self.KEYPOINT_INDICES['left_hip']]
            posture_angles['back_tilt'] = self.calculate_angle(left_shoulder, left_hip, 'vertical')
            
            # Head tilt: angle between left_ear and left_shoulder with vertical
            left_ear = keypoints[self.KEYPOINT_INDICES['left_ear']]
            posture_angles['neck_tilt'] = self.calculate_angle(left_ear, left_shoulder, 'vertical')
            
            # Body tilt: angle between left_shoulder and left_ear with vertical
            posture_angles['body_tilt'] = self.calculate_angle(left_ear, left_hip, 'vertical')
            
            # Hip tilt: angle between left_knee and left_hip with vertical
            left_knee = keypoints[self.KEYPOINT_INDICES['left_knee']]
            posture_angles['hip_tilt'] = self.calculate_angle(left_hip, left_knee, 'vertical')
            
        return posture_angles

    def analyze_posture(self, front_image_data: bytes, side_image_data: bytes) -> PostureAngles:
        """Analyze posture from front and side view images.
        
        Args:
            front_image_data: Raw bytes of front view image
            side_image_data: Raw bytes of side view image
            
        Returns:
            PostureAngles object with calculated angles
            
        Raises:
            ValueError: If pose landmarks cannot be detected in either image
        """
        # Extract keypoints from front image
        front_keypoints = self.extract_keypoints_from_image(front_image_data)
        if front_keypoints is None:
            raise ValueError("Could not detect pose landmarks in front view image")
            
        # Extract keypoints from side image
        side_keypoints = self.extract_keypoints_from_image(side_image_data)
        if side_keypoints is None:
            raise ValueError("Could not detect pose landmarks in side view image")
            
        # Calculate angles for both views
        front_angles = self.calculate_posture_angles(front_keypoints, "front")
        side_angles = self.calculate_posture_angles(side_keypoints, "side")
        
        # Create response objects
        front_view = FrontViewPosture(**front_angles)
        side_view = SideViewPosture(**side_angles)
        
        return PostureAngles(front_view=front_view, side_view=side_view)

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close() 
