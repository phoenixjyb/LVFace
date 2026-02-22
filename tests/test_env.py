#!/usr/bin/env python3
import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

try:
    import onnxruntime
    print(f"✅ ONNX Runtime {onnxruntime.__version__} loaded successfully")
    providers = onnxruntime.get_available_providers()
    print(f"🔧 Available providers: {providers}")
    
    if "CUDAExecutionProvider" in providers:
        print("🚀 GPU support available!")
    else:
        print("❌ No GPU support")
        
except ImportError as e:
    print(f"❌ Failed to import onnxruntime: {e}")

print("🚀 Ready to start service!")
