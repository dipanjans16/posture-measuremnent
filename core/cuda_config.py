"""
CUDA and cuDNN configuration settings
"""

import torch


def configure_cuda():
    """Configure CUDA and cuDNN settings for optimal performance"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Allow TF32 on Ampere
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Don't enforce deterministic algorithms
        torch.backends.cudnn.deterministic = False

        # Set device to GPU
        torch.cuda.set_device(0)  # Use first GPU

        # Clear cache
        torch.cuda.empty_cache()
