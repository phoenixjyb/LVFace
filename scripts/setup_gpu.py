#!/usr/bin/env python3
"""
GPU Setup Script for LVFace
============================

This script helps enable GPU acceleration for LVFace by setting up the CUDA library paths.
"""

import os
import sys
from pathlib import Path

def setup_cuda_environment():
    """Add CUDA library paths to environment for ONNX Runtime GPU support"""
    
    # Find the virtual environment path
    venv_path = Path(sys.executable).parent.parent
    nvidia_path = venv_path / "Lib" / "site-packages" / "nvidia"
    
    if not nvidia_path.exists():
        print("❌ NVIDIA CUDA libraries not found in virtual environment")
        print("Please install with: pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12")
        return False
    
    # Find all CUDA library bin directories
    cuda_bin_paths = []
    for cuda_lib in nvidia_path.glob("*/bin"):
        if cuda_lib.is_dir():
            cuda_bin_paths.append(str(cuda_lib))
    
    if not cuda_bin_paths:
        print("❌ No CUDA bin directories found")
        return False
    
    # Add to PATH
    current_path = os.environ.get("PATH", "")
    new_path = os.pathsep.join(cuda_bin_paths + [current_path])
    os.environ["PATH"] = new_path
    
    print(f"✅ Added {len(cuda_bin_paths)} CUDA library paths to environment:")
    for path in cuda_bin_paths:
        print(f"   - {path}")
    
    return True

def test_gpu_inference():
    """Test if GPU inference is working"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print("\n🧪 Testing GPU inference...")
        
        # Try to create CUDA session
        session = ort.InferenceSession(
            "./models/LVFace-B_Glint360K.onnx",
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        providers = session.get_providers()
        print(f"Active providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("🎉 SUCCESS: GPU acceleration is working!")
            
            # Run a quick benchmark
            dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            
            import time
            start_time = time.time()
            output = session.run(None, {session.get_inputs()[0].name: dummy_input})
            inference_time = time.time() - start_time
            
            print(f"⚡ GPU inference time: {inference_time*1000:.2f}ms")
            print(f"📊 Output shape: {output[0].shape}")
            return True
        else:
            print("⚠️  GPU acceleration not available, using CPU")
            return False
            
    except Exception as e:
        print(f"❌ Error testing GPU inference: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up GPU acceleration for LVFace...\n")
    
    if setup_cuda_environment():
        test_gpu_inference()
    else:
        print("\n❌ Failed to set up GPU acceleration")
        sys.exit(1)
