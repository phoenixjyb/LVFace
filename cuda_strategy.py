#!/usr/bin/env python3
"""
CUDA Installation Strategy Guide
===============================

Comprehensive guide for CUDA setup on Windows with WSL considerations.
"""

def print_cuda_ecosystem():
    """Explain the CUDA ecosystem components"""
    print("🔧 CUDA Ecosystem Components")
    print("=" * 50)
    
    print("\n1️⃣ NVIDIA GPU Driver:")
    print("   • Base layer - required for everything")
    print("   • Provides CUDA Runtime API")
    print("   • Version: Currently 571.59 (supports CUDA 12.8)")
    print("   • Location: System-wide (Windows)")
    
    print("\n2️⃣ CUDA Toolkit (Development):")
    print("   • nvcc compiler, libraries, headers")
    print("   • For compiling CUDA code from source")
    print("   • Size: ~3-4GB")
    print("   • Location: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\")
    
    print("\n3️⃣ CUDA Runtime (Production):")
    print("   • Runtime libraries (.dll files)")
    print("   • For running pre-compiled CUDA applications")
    print("   • Size: ~500MB")
    print("   • Location: Virtual environments or system PATH")
    
    print("\n4️⃣ cuDNN (Deep Learning):")
    print("   • Optimized primitives for neural networks")
    print("   • Required for PyTorch, TensorFlow, ONNX Runtime")
    print("   • Version specific to CUDA version")

def print_installation_strategy():
    """Recommended installation strategy"""
    print("\n🎯 Recommended Installation Strategy")
    print("=" * 50)
    
    print("\n🪟 WINDOWS HOST:")
    print("─" * 20)
    print("✅ DO INSTALL:")
    print("   • NVIDIA GPU Driver (already done ✓)")
    print("   • CUDA Toolkit 12.6 (latest stable)")
    print("   • Visual Studio Build Tools (for nvcc)")
    
    print("\n❌ DON'T INSTALL:")
    print("   • System-wide Python CUDA packages")
    print("   • Multiple CUDA Toolkit versions")
    
    print("\n🐧 WSL2 (if used):")
    print("─" * 20)
    print("✅ DO INSTALL:")
    print("   • WSL2 with Ubuntu 22.04 LTS")
    print("   • CUDA Toolkit in WSL (shares Windows driver)")
    
    print("\n❌ DON'T INSTALL:")
    print("   • Separate GPU drivers in WSL")
    print("   • Conflicting CUDA versions")
    
    print("\n🐍 PYTHON ENVIRONMENTS:")
    print("─" * 20)
    print("✅ PER-PROJECT BASIS:")
    print("   • nvidia-cuda-runtime-cu12 (runtime only)")
    print("   • nvidia-cublas-cu12, nvidia-cudnn-cu12")
    print("   • Isolated in virtual environments")

def print_conflict_avoidance():
    """How to avoid conflicts"""
    print("\n⚠️ Conflict Avoidance Strategy")
    print("=" * 50)
    
    print("\n🎯 GOLDEN RULES:")
    print("1. ONE driver version system-wide")
    print("2. ONE CUDA Toolkit per system (Windows or WSL)")
    print("3. CUDA runtime libraries in venv only")
    print("4. Match CUDA versions (12.x with 12.x)")
    print("5. Use conda OR pip, not both")
    
    print("\n🔄 VERSION COMPATIBILITY:")
    print("   CUDA Driver 12.8 supports:")
    print("   ├── CUDA Toolkit 12.0-12.6")
    print("   ├── cuDNN 8.x and 9.x")
    print("   └── PyTorch CUDA 12.1, 12.4")

def print_installation_steps():
    """Step-by-step installation guide"""
    print("\n📋 Installation Steps")
    print("=" * 50)
    
    print("\n🪟 STEP 1: Windows CUDA Toolkit")
    print("─" * 30)
    print("1. Download: https://developer.nvidia.com/cuda-12-6-0-download")
    print("2. Choose: Windows > x86_64 > 11 > exe (local)")
    print("3. Install: Custom installation")
    print("4. Select: CUDA Toolkit + Visual Studio Integration")
    print("5. Skip: Driver (already have newer)")
    print("6. Verify: nvcc --version")
    
    print("\n🐍 STEP 2: Python Environment Setup")
    print("─" * 30)
    print("# Current working approach - keep using this:")
    print("pip install nvidia-cuda-runtime-cu12")
    print("pip install nvidia-cublas-cu12 nvidia-cudnn-cu12")
    print("pip install onnxruntime-gpu")
    
    print("\n🐧 STEP 3: WSL2 Setup (optional)")
    print("─" * 30)
    print("# In WSL2 Ubuntu:")
    print("wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb")
    print("sudo dpkg -i cuda-keyring_1.0-1_all.deb")
    print("sudo apt-get update")
    print("sudo apt-get install cuda-toolkit-12-6")

def print_verification():
    """How to verify installation"""
    print("\n✅ Verification Commands")
    print("=" * 50)
    
    print("\n🪟 Windows Verification:")
    print("nvcc --version                    # CUDA Toolkit")
    print("nvidia-smi                        # Driver & Runtime")
    print("set CUDA_PATH                     # Environment variable")
    
    print("\n🐍 Python Verification:")
    print("python -c \"import torch; print(torch.cuda.is_available())\"")
    print("python -c \"import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())\"")
    
    print("\n🐧 WSL2 Verification:")
    print("nvcc --version                    # Should match or be < Windows version")
    print("nvidia-smi                        # Should show same GPU")

def print_use_cases():
    """When to use what"""
    print("\n🎯 Use Case Matrix")
    print("=" * 50)
    
    print("\n📊 DEVELOPMENT:")
    print("   • Compiling CUDA kernels → Windows CUDA Toolkit")
    print("   • PyTorch training → Windows + pip packages")
    print("   • Research/experiments → WSL2 + Linux tools")
    
    print("\n🚀 PRODUCTION:")
    print("   • LVFace inference → Windows + pip packages")
    print("   • Web services → WSL2 containers")
    print("   • Batch processing → Windows CUDA Toolkit")
    
    print("\n🔧 TESTING:")
    print("   • Cross-platform → Both Windows & WSL2")
    print("   • CI/CD → Docker with CUDA runtime")

if __name__ == "__main__":
    print("🎯 CUDA Installation Strategy for Windows + WSL2")
    print("=" * 60)
    
    print_cuda_ecosystem()
    print_installation_strategy()
    print_conflict_avoidance()
    print_installation_steps()
    print_verification()
    print_use_cases()
    
    print("\n🎉 RECOMMENDATION FOR YOUR SETUP:")
    print("=" * 40)
    print("1. Install Windows CUDA Toolkit 12.6")
    print("2. Keep current Python venv approach")
    print("3. Add WSL2 CUDA later if needed")
    print("4. RTX 3090 will work perfectly with this setup!")
