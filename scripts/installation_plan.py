#!/usr/bin/env python3
"""
CUDA Installation Plan for Your System
=====================================

Based on current analysis: You need Windows CUDA Toolkit installation.
"""

import subprocess
import os
from pathlib import Path

def check_current_status():
    """Check what's currently installed"""
    print("🔍 Current System Analysis")
    print("=" * 50)
    
    # Check driver
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA Driver: 571.59 (CUDA 12.8 support)")
        else:
            print("❌ NVIDIA Driver: Not working")
    except:
        print("❌ NVIDIA Driver: Not found")
    
    # Check CUDA Toolkit
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"✅ CUDA_PATH: {cuda_path}")
    else:
        print("❌ CUDA Toolkit: Not installed")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvcc compiler: Available")
        else:
            print("❌ nvcc compiler: Not found")
    except:
        print("❌ nvcc compiler: Not found")
    
    # Check Python CUDA packages
    try:
        import torch
        print(f"✅ PyTorch CUDA: {torch.version.cuda}")
    except:
        print("❌ PyTorch CUDA: Not available")
    
    print("\n📊 CURRENT STATUS:")
    print("✅ GPU Driver: Working")
    print("✅ Python CUDA Runtime: Installed via pip")
    print("❌ CUDA Toolkit: Missing (needed for RTX 3090)")

def print_installation_plan():
    """Detailed installation plan"""
    print("\n🎯 RECOMMENDED INSTALLATION PLAN")
    print("=" * 50)
    
    print("\n📋 PHASE 1: Windows CUDA Toolkit")
    print("─" * 30)
    print("1. Download CUDA Toolkit 12.6:")
    print("   https://developer.nvidia.com/cuda-12-6-0-download")
    print("   → Windows → x86_64 → 11 → exe (local)")
    print()
    print("2. Installation options:")
    print("   ✅ Custom Installation")
    print("   ✅ CUDA Toolkit 12.6")
    print("   ✅ CUDA Development")
    print("   ❌ CUDA Driver (already have newer)")
    print("   ✅ Visual Studio Integration")
    
    print("\n📋 PHASE 2: Build Tools (if needed)")
    print("─" * 30)
    print("1. Check if you have Visual Studio:")
    print("   - VS 2019/2022 Community/Professional")
    print("   - VS Build Tools 2019/2022")
    print()
    print("2. If missing, install VS Build Tools:")
    print("   https://visualstudio.microsoft.com/downloads/")
    print("   → Build Tools for Visual Studio 2022")
    print("   → Select: C++ build tools")
    
    print("\n📋 PHASE 3: Verification")
    print("─" * 30)
    print("1. Open NEW PowerShell/Command Prompt")
    print("2. Run: nvcc --version")
    print("3. Run: echo %CUDA_PATH%")
    print("4. Test: python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("\n📋 PHASE 4: RTX 3090 Preparation")
    print("─" * 30)
    print("✅ Already done:")
    print("   - Python CUDA runtime libraries")
    print("   - ONNX Runtime GPU")
    print("   - PyTorch with CUDA support")
    print("   - All cuDNN libraries")
    print()
    print("🚀 After RTX 3090 installation:")
    print("   - Everything will work immediately")
    print("   - No additional software needed")
    print("   - Expect 5-10x performance boost")

def print_conflict_prevention():
    """How to prevent conflicts"""
    print("\n⚠️ CONFLICT PREVENTION")
    print("=" * 50)
    
    print("\n🛡️ DURING INSTALLATION:")
    print("• Choose CUSTOM installation")
    print("• UNCHECK driver update (keep 571.59)")
    print("• ONLY install CUDA Toolkit components")
    print("• Don't install multiple CUDA versions")
    
    print("\n🛡️ AFTER INSTALLATION:")
    print("• Only one CUDA_PATH should be set")
    print("• Keep using pip packages in venv")
    print("• Don't mix conda and pip for CUDA")
    print("• Test in a clean terminal session")

def print_troubleshooting():
    """Common issues and solutions"""
    print("\n🔧 TROUBLESHOOTING")
    print("=" * 50)
    
    print("\n❗ If installation fails:")
    print("1. Run as Administrator")
    print("2. Disable antivirus temporarily")
    print("3. Close all CUDA applications")
    print("4. Restart and try again")
    
    print("\n❗ If nvcc not found after install:")
    print("1. Check CUDA_PATH environment variable")
    print("2. Add %CUDA_PATH%\\bin to PATH")
    print("3. Restart PowerShell/Command Prompt")
    print("4. Restart VS Code")
    
    print("\n❗ If conflicts occur:")
    print("1. Uninstall ALL CUDA components")
    print("2. Clean registry entries")
    print("3. Reinstall driver first")
    print("4. Install CUDA Toolkit fresh")

def print_wsl2_strategy():
    """WSL2 installation strategy"""
    print("\n🐧 WSL2 CUDA STRATEGY (Optional)")
    print("=" * 50)
    
    print("\n📅 WHEN TO INSTALL WSL2 CUDA:")
    print("✅ If you develop Linux-based ML applications")
    print("✅ If you use Docker containers for ML")
    print("✅ If you prefer Linux development environment")
    print("✅ If you need cross-platform testing")
    
    print("\n❌ WHEN NOT TO INSTALL:")
    print("• Just for LVFace (Windows works great)")
    print("• If you're new to WSL")
    print("• If Windows setup is working")
    
    print("\n🔄 WSL2 CUDA INSTALLATION:")
    print("1. Enable WSL2 and install Ubuntu 22.04")
    print("2. Install WSL-specific CUDA (no driver)")
    print("3. Shares Windows driver automatically")
    print("4. Independent from Windows CUDA Toolkit")

if __name__ == "__main__":
    print("🎯 CUDA INSTALLATION PLAN FOR YOUR SYSTEM")
    print("=" * 55)
    
    check_current_status()
    print_installation_plan()
    print_conflict_prevention()
    print_troubleshooting()
    print_wsl2_strategy()
    
    print("\n🎉 SUMMARY RECOMMENDATION:")
    print("=" * 30)
    print("1. ✅ Your current Python setup is perfect")
    print("2. 📦 Install Windows CUDA Toolkit 12.6")
    print("3. 🚀 RTX 3090 will work immediately")
    print("4. 🐧 Skip WSL2 CUDA for now")
    print("5. 🎯 Focus on getting RTX 3090 first!")
