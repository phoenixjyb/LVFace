#!/usr/bin/env python3
try:
    from PIL import Image
    print("✅ PIL Image imported successfully")
except ImportError as e:
    print(f"❌ PIL Image import failed: {e}")

try:
    import onnxruntime
    print("✅ ONNX Runtime available")
except ImportError as e:
    print(f"❌ ONNX Runtime failed: {e}")

try:
    from flask import Flask
    print("✅ Flask available")
except ImportError as e:
    print(f"❌ Flask failed: {e}")

print("🚀 Ready to start LVFace service!")
