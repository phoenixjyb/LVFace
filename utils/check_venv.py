#!/usr/bin/env python3

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("\nSite-packages paths:")
for p in sys.path:
    if "site-packages" in p:
        print(f"  {p}")

print("\nTrying to import onnxruntime...")
try:
    import onnxruntime as ort
    print("✅ onnxruntime imported successfully")
    print(f"ONNX file location: {ort.__file__}")
    
    # Try to check version
    try:
        print(f"ONNX version: {ort.__version__}")
    except AttributeError:
        print("⚠️ No __version__ attribute")
    
    # Try to check providers
    try:
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
    except AttributeError:
        print("⚠️ No get_available_providers() method")
        
    try:
        providers = ort.get_all_providers()
        print(f"All providers: {providers}")
    except AttributeError:
        print("⚠️ No get_all_providers() method")
        
except ImportError as e:
    print(f"❌ Failed to import onnxruntime: {e}")

print("\nChecking installed packages...")
import subprocess
try:
    result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        onnx_packages = [line for line in lines if 'onnx' in line.lower()]
        if onnx_packages:
            print("ONNX-related packages:")
            for pkg in onnx_packages:
                print(f"  {pkg}")
        else:
            print("No ONNX packages found in pip list")
    else:
        print(f"Pip list failed: {result.stderr}")
except Exception as e:
    print(f"Error checking packages: {e}")
