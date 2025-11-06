#!/bin/bash
# Setup script for PNR Graphs project
# This script creates a conda environment and installs all dependencies

set -e  # Exit on error

ENV_NAME="pnr-graphs"
PYTHON_VERSION="3.10"

echo "Setting up PNR Graphs environment..."

# Create conda environment (skip if already exists)
echo "Creating conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo " Environment $ENV_NAME already exists. Skipping creation."
else
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi

# Activate environment and install packages
echo "Installing packages..."
conda run -n $ENV_NAME pip install --upgrade pip

# Detect OS and install appropriate PyTorch version
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    # macOS - use CPU version (Metal support available but CPU is more compatible)
    echo "Detected macOS. Installing PyTorch (CPU)..."
    conda run -n $ENV_NAME pip install torch torchvision torchaudio
    echo "Installing PyTorch Geometric..."
    # On macOS, torch-scatter/sparse may need compilation or may not be available
    # Install torch-geometric first, which will handle dependencies
    conda run -n $ENV_NAME pip install torch-geometric || echo " torch-geometric installation had issues, continuing..."
else
    # Linux - use CUDA 12.1 (adjust if needed)
    echo "Detected Linux. Installing PyTorch (CUDA 12.1)..."
    conda run -n $ENV_NAME pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    echo "Installing PyTorch Geometric (CUDA 12.1)..."
    conda run -n $ENV_NAME pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
fi

# Install core libraries
echo "Installing core libraries..."
conda run -n $ENV_NAME pip install opencv-python numpy scipy scikit-learn networkx tqdm addict yacs

# Install vision stacks
echo "Installing vision libraries..."
conda run -n $ENV_NAME pip install groundingdino-py supervision pillow

# Clone and install GroundingDINO
echo "Installing GroundingDINO..."
if [ ! -d "GroundingDINO" ]; then
    git clone https://github.com/IDEA-Research/GroundingDINO.git
fi
# Install GroundingDINO dependencies first (if requirements.txt exists)
if [ -f "GroundingDINO/requirements.txt" ]; then
    echo "   Installing GroundingDINO dependencies..."
    conda run -n $ENV_NAME pip install -r GroundingDINO/requirements.txt || echo "   Some dependencies may have failed, continuing..."
fi
# Install GroundingDINO in editable mode
# Use --no-build-isolation to use already installed torch
echo "   Installing GroundingDINO package..."
conda run -n $ENV_NAME pip install --no-build-isolation -e GroundingDINO || {
    echo "   Retrying without --no-build-isolation..."
    conda run -n $ENV_NAME pip install -e GroundingDINO
}

# Clone and install SAM2
echo "Installing SAM2..."
if [ ! -d "sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git
fi
# Install SAM2 dependencies first (if requirements.txt exists)
if [ -f "sam2/requirements.txt" ]; then
    echo "   Installing SAM2 dependencies..."
    conda run -n $ENV_NAME pip install -r sam2/requirements.txt || echo "   Some dependencies may have failed, continuing..."
fi
# Install SAM2 in editable mode
echo "   Installing SAM2 package..."
conda run -n $ENV_NAME pip install --no-build-isolation -e ./sam2 || {
    echo "   Retrying without --no-build-isolation..."
    conda run -n $ENV_NAME pip install -e ./sam2
}

echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"

