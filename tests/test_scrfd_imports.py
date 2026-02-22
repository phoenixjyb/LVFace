#!/usr/bin/env python3
"""Test typing_extensions and InsightFace imports"""

print("Testing imports...")

try:
    import typing_extensions
    print("✅ typing_extensions imported successfully")
    print(f"   Version: {typing_extensions.__version__}")
except Exception as e:
    print(f"❌ typing_extensions import failed: {e}")

try:
    import insightface
    print("✅ InsightFace imported successfully")
    print(f"   Version: {insightface.__version__}")
except Exception as e:
    print(f"❌ InsightFace import failed: {e}")

try:
    from insightface.app import FaceAnalysis
    print("✅ FaceAnalysis imported successfully")
    
    # Test SCRFD initialization
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✅ FaceAnalysis created successfully")
    
    # Test prepare (this downloads models)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ SCRFD models prepared successfully")
    
except Exception as e:
    print(f"❌ FaceAnalysis/SCRFD failed: {e}")
    import traceback
    traceback.print_exc()
