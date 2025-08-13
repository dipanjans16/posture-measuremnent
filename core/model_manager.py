"""
Model manager for SMPL-X model initialization and management.
"""

import argparse
import os
import os.path as osp
import sys
from typing import Optional, Dict

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from fastapi import HTTPException

# Add the main and data directories to Python path BEFORE importing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
main_path = osp.join(project_root, "main")
data_path = osp.join(project_root, "data")

sys.path.insert(0, main_path)
sys.path.insert(0, data_path)

# Now import the original config module
from config import cfg
from common.base import Demoer
from app_config.settings import APIConfig


class ModelManager:
    """Manages SMPL-X model initialization and configuration."""

    def __init__(self):
        # self._demoer: Optional[Demoer] = None # Old: single demoer
        self._demoers: Dict[str, Demoer] = {}
        self._transform: Optional[transforms.ToTensor] = None
        self._model_initialized: bool = False

    def initialize_model(self):
        """Initialize the SMPL-X model and related components for all supported sexes."""
        if self._model_initialized:
            return

        # Set default arguments for OSX model (encoder/decoder network)
        parser = argparse.ArgumentParser()
        parser.add_argument("--gpu", type=str, default=APIConfig.DEFAULT_GPU)
        parser.add_argument(
            "--encoder_setting", type=str, default=APIConfig.DEFAULT_ENCODER_SETTING
        )
        parser.add_argument(
            "--decoder_setting", type=str, default=APIConfig.DEFAULT_DECODER_SETTING
        )
        parser.add_argument(
            "--pretrained_model_path", type=str, default=APIConfig.DEFAULT_MODEL_PATH
        )
        args = parser.parse_args([])

        # Initialize configuration for OSX model
        cfg.set_args(args.gpu)
        cudnn.benchmark = True

        cfg.set_additional_args(
            encoder_setting=args.encoder_setting,
            decoder_setting=args.decoder_setting,
            pretrained_model_path=args.pretrained_model_path,
        )

        # Initialize demoer for each supported sex
        for sex in APIConfig.SUPPORTED_SEXES:
            sex_lower = sex.lower()
            print(f"Initializing model for sex: {sex_lower}...")
            try:
                demoer_instance = Demoer(
                    sex=sex_lower
                )  # Pass sex to Demoer constructor
                demoer_instance._make_model()  # This loads the OSX network and sets up SMPLX layer via OSX.Model
                demoer_instance.model.eval()
                self._demoers[sex_lower] = demoer_instance
                print(f"Successfully initialized model for sex: {sex_lower}")
            except Exception as e:
                print(f"Error initializing model for sex {sex_lower}: {e}")
                # Depending on policy, either raise error or continue without this model
                # For now, let's be strict and raise if any model fails, or make it configurable
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to initialize model for sex {sex_lower}. Check model files and configurations. Error: {e}",
                )

        if not self._demoers:
            raise HTTPException(
                status_code=500,
                detail="No models were initialized. Check APIConfig.SUPPORTED_SEXES and model availability.",
            )

        # Initialize transform (common for all)
        self._transform = transforms.ToTensor()

        self._model_initialized = True
        print(
            f"All models initialized successfully for sexes: {list(self._demoers.keys())}"
        )

    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._model_initialized and len(self._demoers) > 0

    def get_demoer(self, sex: str) -> Demoer:
        """Get the demoer instance for a specific sex."""
        if not self.is_initialized():
            raise HTTPException(status_code=500, detail="Models not initialized")
        sex_lower = sex.lower()
        if sex_lower not in self._demoers:
            raise HTTPException(
                status_code=400,
                detail=f"Model for sex '{sex_lower}' is not available or not supported. Supported: {list(self._demoers.keys())}",
            )
        return self._demoers[sex_lower]

    def get_transform(self) -> transforms.ToTensor:
        """Get the transform instance."""
        if not self._model_initialized:
            raise HTTPException(status_code=500, detail="Model not initialized")
        return self._transform


# Create a global instance
model_manager = ModelManager()
