"""
Pydantic schemas for Body Measurement API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class FrontViewPosture(BaseModel):
    """Front view posture analysis results."""
    
    shoulder_tilt: float = Field(
        ..., 
        description="Shoulder tilt angle in degrees relative to horizontal"
    )
    pelvic_tilt: float = Field(
        ..., 
        description="Pelvic tilt angle in degrees relative to horizontal"
    )
    head_tilt: float = Field(
        ..., 
        description="Head tilt angle in degrees relative to horizontal"
    )
    body_tilt: float = Field(
        ..., 
        description="Body tilt angle in degrees relative to vertical"
    )


class SideViewPosture(BaseModel):
    """Side view posture analysis results."""
    
    back_tilt: float = Field(
        ..., 
        description="Back tilt angle between shoulder and hip relative to vertical"
    )
    neck_tilt: float = Field(
        ..., 
        description="Neck tilt angle between ear and shoulder relative to vertical"  
    )
    body_tilt: float = Field(
        ..., 
        description="Overall body tilt angle from ear to hip relative to vertical"
    )
    hip_tilt: float = Field(
        ..., 
        description="Hip tilt angle between hip and knee relative to vertical"
    )


class PostureAngles(BaseModel):
    """Complete posture analysis from front and side views."""
    
    front_view: FrontViewPosture = Field(
        ..., 
        description="Posture measurements from front view"
    )
    side_view: SideViewPosture = Field(
        ..., 
        description="Posture measurements from side view"
    )
