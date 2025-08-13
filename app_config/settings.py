"""
Configuration settings for the Posture Measurement API
"""

import os


class APIConfig:
    """API Configuration settings."""

    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    TITLE = "Posture Measurement API"
    VERSION = "1.0.0"
    DESCRIPTION = "AI-powered posture analysis service using MediaPipe pose detection"

    # File upload settings
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
