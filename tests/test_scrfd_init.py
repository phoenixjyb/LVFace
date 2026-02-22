#!/usr/bin/env python3
"""Test SCRFD initialization step by step"""

print("🚀 Testing SCRFD initialization...")

try:
    print("1. Importing InsightFace...")
    import insightface
    print("✅ InsightFace imported")
    
    print("2. Importing FaceAnalysis...")
    from insightface.app import FaceAnalysis
    print("✅ FaceAnalysis imported")
    
    print("3. Creating FaceAnalysis instance...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✅ FaceAnalysis instance created")
    
    print("4. Preparing models (this may download models)...")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ SCRFD models prepared successfully!")
    
    print("🎉 SCRFD is ready to use!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
