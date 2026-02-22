#!/usr/bin/env python3
"""Manual SCRFD model download"""

import os
import urllib.request
from pathlib import Path
import zipfile
import time

def download_scrfd_models():
    """Manually download SCRFD buffalo_l models"""
    
    # Setup paths
    home_dir = Path.home()
    models_dir = home_dir / ".insightface" / "models"
    buffalo_dir = models_dir / "buffalo_l"
    zip_path = models_dir / "buffalo_l.zip"
    
    print(f"🏠 Home directory: {home_dir}")
    print(f"📁 Models directory: {models_dir}")
    print(f"🎯 Target directory: {buffalo_dir}")
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory: {models_dir}")
    
    # Download URL
    download_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    print(f"📥 Downloading from: {download_url}")
    print("   This may take several minutes...")
    
    try:
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r📊 Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(download_url, zip_path, progress_hook)
        print()  # New line after progress
        print(f"✅ Download complete: {zip_path}")
        
        # Extract the zip file
        print("📦 Extracting models...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        print(f"✅ Models extracted to: {buffalo_dir}")
        
        # List extracted files
        if buffalo_dir.exists():
            onnx_files = list(buffalo_dir.glob("*.onnx"))
            print(f"🎯 ONNX model files ({len(onnx_files)}):")
            for onnx_file in onnx_files:
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                print(f"   - {onnx_file.name}: {size_mb:.1f} MB")
        
        # Cleanup zip file
        zip_path.unlink()
        print("🗑️ Cleaned up zip file")
        
        print("🎉 SCRFD models successfully downloaded and installed!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    download_scrfd_models()
