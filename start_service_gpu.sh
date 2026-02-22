#!/bin/bash

echo "🚀 Starting LVFace Service with RTX 3090 GPU Acceleration"
echo "=" * 60

# Navigate to LVFace directory
cd /mnt/c/Users/yanbo/wSpace/vlm-photo-engine/LVFace

# Kill any existing processes
pkill -f "python.*inference.py" 2>/dev/null || echo "No existing service found"

# Check GPU status
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | grep RTX

echo ""
echo "🔧 Starting LVFace service..."

# Start the service with GPU acceleration
source .venv-cuda124-wsl/bin/activate
python3 inference.py

echo "🚀 LVFace service started successfully!"
