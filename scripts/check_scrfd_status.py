#!/usr/bin/env python3
"""Check SCRFD download status"""

import os
from pathlib import Path

def check_scrfd_status():
    """Check current SCRFD model download status"""
    
    home_dir = Path.home()
    insightface_dir = home_dir / ".insightface"
    models_dir = insightface_dir / "models"
    buffalo_dir = models_dir / "buffalo_l"
    
    print(f"🔍 Checking SCRFD status...")
    print(f"   Home directory: {home_dir}")
    print(f"   InsightFace directory: {insightface_dir}")
    
    if not insightface_dir.exists():
        print("❌ InsightFace directory does not exist")
        return False
    
    if not models_dir.exists():
        print("❌ Models directory does not exist")
        return False
        
    print(f"✅ Models directory exists: {models_dir}")
    
    # List all files in models directory
    if models_dir.exists():
        all_files = list(models_dir.rglob("*"))
        print(f"📁 Files in models directory ({len(all_files)} total):")
        for file_path in sorted(all_files):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                relative_path = file_path.relative_to(models_dir)
                print(f"   - {relative_path}: {size_mb:.1f} MB")
    
    # Check buffalo_l specifically
    if buffalo_dir.exists():
        print(f"✅ Buffalo_l directory exists: {buffalo_dir}")
        onnx_files = list(buffalo_dir.glob("*.onnx"))
        print(f"🎯 ONNX model files: {len(onnx_files)}")
        for onnx_file in onnx_files:
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            print(f"   - {onnx_file.name}: {size_mb:.1f} MB")
        
        if len(onnx_files) >= 4:  # Buffalo_l should have 4+ ONNX files
            print("🎉 SCRFD models appear to be complete!")
            return True
        else:
            print("⚠️ SCRFD models appear incomplete")
            return False
    else:
        print("❌ Buffalo_l directory does not exist")
        return False

if __name__ == "__main__":
    check_scrfd_status()
