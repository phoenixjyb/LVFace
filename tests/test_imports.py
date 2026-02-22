#!/usr/bin/env python3
try:
    import flask
    print(f"✅ Flask {flask.__version__} imported successfully")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    from flask import Flask, request, jsonify
    print("✅ Flask components imported successfully")
except ImportError as e:
    print(f"❌ Flask components import failed: {e}")

try:
    import onnxruntime
    print(f"✅ ONNX Runtime {onnxruntime.__version__} imported successfully")
except ImportError as e:
    print(f"❌ ONNX Runtime import failed: {e}")

print("🚀 All imports test complete!")
