#!/usr/bin/env python3
"""Test InsightFace installation and basic functionality"""

import sys
import os

try:
    import insightface
    print("✅ InsightFace imported successfully!")
    print(f"InsightFace version: {insightface.__version__}")
    
    # Test FaceAnalysis creation
    app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✅ FaceAnalysis created successfully!")
    
    # Test model preparation (this downloads models if needed)
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ Models prepared successfully!")
    
    print("🎉 InsightFace is ready for use!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
