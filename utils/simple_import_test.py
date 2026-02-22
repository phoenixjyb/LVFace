#!/usr/bin/env python3
"""Simple test for InsightFace imports"""

try:
    import insightface
    print("✅ InsightFace imported successfully")
except Exception as e:
    print(f"❌ InsightFace import failed: {e}")
    exit(1)

try:
    from insightface.app import FaceAnalysis
    print("✅ FaceAnalysis imported successfully")
except Exception as e:
    print(f"❌ FaceAnalysis import failed: {e}")
    exit(1)

print("🎉 All imports successful!")
