#!/usr/bin/env python3
"""
Body Measurement API Setup Script

This script automates the virtual environment creation and dependency installation
for the Body Measurement API project.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_step(step_num, description):
    """Print formatted step information."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")


def run_command(command, description, shell=False):
    """Run a command and handle errors."""
    print(f"\n> {description}")
    print(f"Running: {command}")
    
    try:
        # Always use shell=True on Windows to handle paths with spaces correctly
        if platform.system() == "Windows" or shell:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        
        print(f"✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}")
        print(f"Error message: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.10"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    if (version.major < 3) or (version.minor != 10):
        print(f"✗ Python 3.10 required. Current version: {version.major}.{version.minor}")
        print("Please install Python 3.10 from https://python.org if not already installed")
        return False
    
    print(f"✓ Python {version.major}.{version.minor} detected")
    return True


def create_virtual_environment():
    """Create virtual environment."""
    print_step(2, "Creating Virtual Environment")
    
    venv_name = "venv"
    
    if os.path.exists(venv_name):
        print(f"Virtual environment '{venv_name}' already exists.")
        response = input("Do you want to recreate it? (y/n): ").lower().strip()
        if response == 'y':
            print("Removing existing virtual environment...")
            if platform.system() == "Windows":
                run_command(f"rmdir /s /q {venv_name}", "Removing old environment", shell=True)
            else:
                run_command(f"rm -rf {venv_name}", "Removing old environment")
        else:
            print("Using existing virtual environment.")
            return True
    
    return run_command(f"python -m venv {venv_name}", "Creating virtual environment")


def get_activation_command():
    """Get the correct activation command for the platform."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"


def install_dependencies():
    """Install project dependencies."""
    print_step(3, "Installing Dependencies")
    
    # Get pip path
    if platform.system() == "Windows":
        pip_path = "venv\\Scripts\\pip"
        mim_path = "venv\\Scripts\\mim"
        python_path = "venv\\Scripts\\python"
    else:
        pip_path = "venv/bin/pip"
        mim_path = "venv/bin/mim"
        python_path = "venv/bin/python"
    
    # Update pip first
    if not run_command(f"{python_path} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install PyTorch with CUDA support
    pytorch_cmd = f"{pip_path} install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113"
    if not run_command(pytorch_cmd, "Installing PyTorch with CUDA support"):
        return False
    
    # Install other dependencies
    deps = [
        (f"{pip_path} install openmim", "Installing OpenMIM"),
        (f"{mim_path} install mmcv-full==1.7.1", "Installing MMCV"),
        (f"{pip_path} install -r requirements.txt", "Installing requirements"),
        (f"{pip_path} install ffmpeg-python", "Installing ffmpeg"),
    ]
    
    for cmd, desc in deps:
        if not run_command(cmd, desc):
            return False
    
    # Install transformer utils
    original_dir = os.getcwd()
    os.chdir(original_dir)
    import subprocess

    # Run the combined command
    result = subprocess.run(
        'cd main/transformer_utils && python setup.py install',
        shell=True,
        capture_output=True,
        text=True
    )
    
    return True


def print_completion_message():
    """Print setup completion message."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    activation_cmd = get_activation_command()
    
    print("\nNext steps:")
    print(f"1. Activate the virtual environment: {activation_cmd}")
    print("2. Start the API server: python main.py")
    print("3. Open your browser to: http://localhost:8000/docs")
    print("4. Test with images in demo_images/ folder")

def main():
    """Main setup function."""
    print("Body Measurement API - Automated Setup")
    print("This script will set up your development environment.")
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("✗ Error: Please run this script from the project root directory")
        print("The directory should contain main.py, requirements.txt, etc.")
        sys.exit(1)
    
    # Run setup steps
    steps = [
        check_python_version,
        create_virtual_environment,
        install_dependencies,
    ]
    
    for step in steps:
        if not step():
            print(f"\n✗ Setup failed at: {step.__name__}")
            print("Please check the error messages above and try again.")
            sys.exit(1)
            
    print_completion_message()


if __name__ == "__main__":
    main() 
