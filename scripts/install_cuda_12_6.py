#!/usr/bin/env python3
"""
CUDA 12.6 Installation Guide
============================

Step-by-step guide to install CUDA 12.6 for RTX 3090 preparation.
"""

import subprocess
import sys
import os
import webbrowser

def check_prerequisites():
    """Check system prerequisites"""
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 50)
    
    print("\n✅ Current Status:")
    print("• NVIDIA Driver: 571.59 (supports CUDA up to 12.8)")
    print("• Python Environment: 3.13 virtual environment")
    print("• PyTorch: 2.6.0+cu124 (CUDA 12.4 compatible)")
    print("• ONNX Runtime: 1.22.0 with GPU support")
    print("• Target GPU: RTX 3090 (when available)")
    
    print("\n🎯 Installation Target:")
    print("• CUDA Toolkit 12.6")
    print("• Custom installation (no driver update)")
    print("• Development tools included")

def download_instructions():
    """Provide download instructions"""
    print("\n📥 DOWNLOAD CUDA 12.6")
    print("=" * 50)
    
    print("\n🌐 Download URL:")
    url = "https://developer.nvidia.com/cuda-12-6-0-download-archive"
    print(f"   {url}")
    
    print("\n📋 Selection Steps:")
    print("1. Operating System: Windows")
    print("2. Architecture: x86_64")
    print("3. Version: 11")
    print("4. Installer Type: exe (local)")
    
    print("\n📦 File Details:")
    print("• Filename: cuda_12.6.0_560.76_windows.exe")
    print("• Size: ~3.5 GB")
    print("• Type: Local installer (offline)")
    
    # Ask if user wants to open the download page
    print("\n🔗 Would you like me to open the download page?")
    response = input("   Press Enter to open, or 'n' to skip: ").strip().lower()
    
    if response != 'n':
        try:
            webbrowser.open(url)
            print("✅ Opening download page in browser...")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print(f"   Please manually visit: {url}")

def installation_steps():
    """Provide detailed installation steps"""
    print("\n🛠️  INSTALLATION STEPS")
    print("=" * 50)
    
    print("\n📋 Step 1: Run Installer")
    print("• Right-click cuda_12.6.0_560.76_windows.exe")
    print("• Select 'Run as administrator'")
    print("• Wait for extraction and initialization")
    
    print("\n📋 Step 2: Installation Type")
    print("• Choose: 'Custom (Advanced)'")
    print("• DO NOT choose 'Express'")
    print("• This gives you control over components")
    
    print("\n📋 Step 3: Component Selection")
    print("✅ INCLUDE:")
    print("   • CUDA Toolkit 12.6")
    print("   • CUDA Development Tools")
    print("   • CUDA Runtime Libraries")
    print("   • CUDA Documentation (optional)")
    print("   • CUDA Samples (optional)")
    
    print("\n❌ EXCLUDE:")
    print("   • NVIDIA Display Driver")
    print("   • PhysX System Software")
    print("   • GeForce Experience components")
    print("   • NVIDIA Graphics Driver")
    
    print("\n📋 Step 4: Installation Path")
    print("• Default: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6")
    print("• Keep default unless you have specific requirements")
    
    print("\n📋 Step 5: Complete Installation")
    print("• Click 'Install'")
    print("• Wait for installation (10-15 minutes)")
    print("• Reboot when prompted")

def post_installation_verification():
    """Steps to verify installation"""
    print("\n✅ POST-INSTALLATION VERIFICATION")
    print("=" * 50)
    
    print("\n🔧 Step 1: Environment Variables")
    print("Check these are automatically added:")
    print("• CUDA_PATH = C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6")
    print("• PATH includes: %CUDA_PATH%\\bin")
    print("• PATH includes: %CUDA_PATH%\\libnvvp")
    
    print("\n🔧 Step 2: Command Line Test")
    print("Open new PowerShell and run:")
    print("   nvcc --version")
    print("   nvidia-smi")
    
    print("\n🔧 Step 3: Python Integration Test")
    print("We'll test this after installation with your LVFace setup")

def what_happens_next():
    """Explain next steps after installation"""
    print("\n🚀 WHAT HAPPENS NEXT")
    print("=" * 50)
    
    print("\n📊 Immediate Benefits:")
    print("• CUDA development environment ready")
    print("• Prepared for RTX 3090 installation")
    print("• Better PyTorch GPU utilization")
    
    print("\n🎯 After RTX 3090 Installation:")
    print("• Run our demo_gpu.py for GPU testing")
    print("• Expected LVFace performance: ~5-10ms inference")
    print("• 5-10x speedup over current CPU performance")
    
    print("\n⚡ Performance Expectations:")
    print("• Current (CPU): ~54ms per inference")
    print("• Future (RTX 3090): ~5-10ms per inference")
    print("• Throughput: ~100-200 FPS vs current 18 FPS")

def troubleshooting_tips():
    """Common issues and solutions"""
    print("\n🔧 TROUBLESHOOTING TIPS")
    print("=" * 50)
    
    print("\n❓ If Installation Fails:")
    print("• Ensure you have admin privileges")
    print("• Temporarily disable antivirus")
    print("• Check disk space (need ~10GB free)")
    print("• Run installer compatibility troubleshooter")
    
    print("\n❓ If nvcc Command Not Found:")
    print("• Reboot after installation")
    print("• Check environment variables manually")
    print("• Add CUDA paths to PATH manually if needed")
    
    print("\n❓ If PyTorch Doesn't See CUDA:")
    print("• Restart Python environment")
    print("• Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("• May need to reinstall PyTorch after CUDA installation")

def installation_checklist():
    """Final checklist before starting"""
    print("\n📝 PRE-INSTALLATION CHECKLIST")
    print("=" * 50)
    
    checklist = [
        "✅ NVIDIA Driver 571.59 installed and working",
        "✅ At least 10GB free disk space",
        "✅ Administrator privileges available",
        "✅ Antivirus temporarily disabled (optional)",
        "✅ No other CUDA versions currently installed",
        "✅ Ready to reboot after installation"
    ]
    
    for item in checklist:
        print(f"   {item}")
    
    print("\n🎯 Ready to proceed with installation!")

if __name__ == "__main__":
    print("🚀 CUDA 12.6 INSTALLATION GUIDE")
    print("=" * 50)
    
    check_prerequisites()
    download_instructions()
    installation_steps()
    post_installation_verification()
    what_happens_next()
    troubleshooting_tips()
    installation_checklist()
    
    print("\n" + "=" * 50)
    print("🎉 Follow these steps to install CUDA 12.6!")
    print("🔄 Run this script again if you need to reference the steps.")
    print("=" * 50)
