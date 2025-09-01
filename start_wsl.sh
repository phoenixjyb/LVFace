#!/bin/bash
# This script runs the LVFace inference service inside WSL.

# Define the path to the project directory
PROJECT_DIR=$(dirname "$(readlink -f "$0")")
VENV_DIR="$PROJECT_DIR/.venv-wsl"

echo "--- Starting LVFace Service in WSL ---"
echo "Project Directory: $PROJECT_DIR"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Python virtual environment not found at $VENV_DIR."
    echo "Please create it first by running the following commands in your WSL terminal:"
    echo "1. cd \"$PROJECT_DIR\""
    echo "2. python3 -m venv .venv-wsl"
    echo "3. source .venv-wsl/bin/activate"
    echo "4. pip install -r requirements.txt"
    exit 1
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Python environment activated."
echo "Running ONNX inference script..."

# Run the inference script
# We pass --device cuda:0 to ensure it uses the GPU
python "$PROJECT_DIR/inference_onnx.py" --device cuda:0
