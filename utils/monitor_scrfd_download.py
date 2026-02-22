#!/usr/bin/env python3
"""
Monitor SCRFD model download progress
"""

import os
import time
import subprocess
import threading
from pathlib import Path

def monitor_download_directory():
    """Monitor the InsightFace download directory for progress"""
    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    download_dir = Path.home() / ".insightface" / "models"
    
    print(f"🔍 Monitoring directory: {download_dir}")
    
    # Check if models already exist
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"✅ SCRFD models already exist in {model_dir}")
        model_files = list(model_dir.glob("*.onnx"))
        print(f"   Found {len(model_files)} ONNX model files:")
        for model_file in model_files:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"     - {model_file.name}: {size_mb:.1f} MB")
        return True
    
    print(f"📦 Models not found, will monitor download...")
    
    # Monitor for download activity
    start_time = time.time()
    last_size = 0
    check_count = 0
    
    while check_count < 120:  # Monitor for up to 2 minutes
        try:
            # Check for zip file being downloaded
            zip_files = list(download_dir.glob("*.zip"))
            partial_files = list(download_dir.glob("*.zip.part"))
            tmp_files = list(download_dir.glob("*.tmp"))
            
            if zip_files or partial_files or tmp_files:
                all_files = zip_files + partial_files + tmp_files
                current_size = sum(f.stat().st_size for f in all_files if f.exists())
                
                if current_size > last_size:
                    size_mb = current_size / (1024 * 1024)
                    elapsed = time.time() - start_time
                    speed = (current_size - last_size) / (1024 * 1024) if elapsed > 0 else 0
                    
                    print(f"📥 Downloading: {size_mb:.1f} MB ({speed:.1f} MB/s)")
                    last_size = current_size
                
            # Check if models directory was created
            if model_dir.exists():
                model_files = list(model_dir.glob("*.onnx"))
                if model_files:
                    print(f"✅ Download complete! Found {len(model_files)} model files:")
                    for model_file in model_files:
                        size_mb = model_file.stat().st_size / (1024 * 1024)
                        print(f"   - {model_file.name}: {size_mb:.1f} MB")
                    return True
            
            time.sleep(1)
            check_count += 1
            
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(1)
            check_count += 1
    
    print("⏰ Download monitoring timeout")
    return False

def test_scrfd_with_progress():
    """Test SCRFD initialization with progress monitoring"""
    
    print("🚀 Testing SCRFD with download progress monitoring")
    
    # Start download monitoring in background
    monitor_thread = threading.Thread(target=monitor_download_directory)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        print("1. Importing InsightFace...")
        import insightface
        from insightface.app import FaceAnalysis
        print("✅ InsightFace imports successful")
        
        print("2. Creating FaceAnalysis instance...")
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print("✅ FaceAnalysis instance created")
        
        print("3. Preparing models (this triggers download if needed)...")
        start_time = time.time()
        
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        end_time = time.time()
        print(f"✅ SCRFD models prepared in {end_time - start_time:.1f} seconds!")
        
        # Test basic functionality
        print("4. Testing SCRFD detection...")
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(test_image)
        print(f"✅ SCRFD test successful! Detected {len(faces)} faces in test image")
        
        print("🎉 SCRFD is ready for use!")
        return True
        
    except Exception as e:
        print(f"❌ SCRFD test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_scrfd_with_progress()
