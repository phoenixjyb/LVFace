#!/bin/bash
# Start the unified SCRFD + LVFace service

cd /mnt/c/Users/yanbo/wSpace/vlm-photo-engine/LVFace

echo "🚀 Starting Unified SCRFD + LVFace Service..."

# Kill any existing service
pkill -f unified_scrfd_service.py

# Start the service
.venv-cuda124-wsl/bin/python unified_scrfd_service.py
