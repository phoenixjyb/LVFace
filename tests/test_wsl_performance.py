#!/usr/bin/env python3

import requests
import sqlite3
import json
import time
import os
import base64
import subprocess
from datetime import datetime

def test_face_inference_in_wsl():
    """Test face inference directly inside WSL to bypass proxy issues"""
    
    print("🧪 TESTING LVFace SERVICE INSIDE WSL (BYPASS PROXY)")
    print("=" * 60)
    
    # Check service health from inside WSL
    try:
        response = requests.get("http://127.0.0.1:8003/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Service Health (from WSL):")
            for key, value in health_data.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    # Get test image from Windows path (accessible via /mnt/c)
    print("\n📸 Using test image...")
    
    # Find a test image in the database path
    test_image_path = "/mnt/e/01_INCOMING/Jane/20220112_043621.jpg"  # From previous database query
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("📂 Let's find an available image...")
        
        # Try to find any image in the incoming directory
        try:
            result = subprocess.run(['find', '/mnt/e/01_INCOMING/', '-name', '*.jpg', '-type', 'f'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                available_images = result.stdout.strip().split('\n')[:3]
                test_image_path = available_images[0]
                print(f"📸 Found image: {test_image_path}")
            else:
                print("❌ No images found")
                return
        except Exception as e:
            print(f"❌ Error finding images: {e}")
            return
    
    print(f"📁 Test image: {test_image_path}")
    
    # Test the /embed endpoint with GPU monitoring
    print(f"\n🔍 Testing /embed endpoint with RTX 3090 monitoring...")
    
    inference_times = []
    num_tests = 3
    
    for i in range(num_tests):
        try:
            print(f"\n  📸 Test {i+1}/{num_tests}: {os.path.basename(test_image_path)}")
            
            # Get GPU stats before inference
            gpu_before = get_gpu_stats()
            
            # Read and encode image to base64
            with open(test_image_path, 'rb') as img_file:
                image_data = img_file.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Start timing
            start_time = time.time()
            
            # Send request to /embed endpoint
            response = requests.post(
                "http://127.0.0.1:8003/embed",
                json={'image': image_b64},
                timeout=30
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Get GPU stats after inference
            gpu_after = get_gpu_stats()
            
            if response.status_code == 200:
                result_data = response.json()
                inference_times.append(inference_time)
                
                print(f"    ✅ SUCCESS! Time: {inference_time:.4f}s")
                
                # Show embedding info
                if 'embedding' in result_data:
                    embedding = result_data['embedding']
                    print(f"    🧠 Embedding size: {len(embedding)}")
                    print(f"    📐 Embedding shape: {result_data.get('shape', 'N/A')}")
                
                # Performance assessment
                if inference_time < 0.02:
                    print("    🚀 BLAZING FAST: RTX 3090 at maximum power!")
                elif inference_time < 0.05:
                    print("    ⚡ VERY FAST: Excellent GPU acceleration")
                elif inference_time < 0.1:
                    print("    ✅ FAST: Good GPU performance")
                elif inference_time < 0.3:
                    print("    ⚠️  MODERATE: Some GPU usage")
                else:
                    print("    🐌 SLOW: Possible CPU fallback")
                
                # Show GPU utilization during inference
                if gpu_before and gpu_after:
                    gpu_util_before = float(gpu_before.split(',')[1]) if len(gpu_before.split(',')) > 1 else 0
                    gpu_util_after = float(gpu_after.split(',')[1]) if len(gpu_after.split(',')) > 1 else 0
                    print(f"    🖥️  GPU util: {gpu_util_before}% → {gpu_util_after}%")
                    
            else:
                print(f"    ❌ Error {response.status_code}: {response.text[:100]}...")
                
        except Exception as e:
            print(f"    ❌ Exception: {str(e)[:100]}...")
        
        # Small delay between tests
        time.sleep(1)
    
    # Results summary
    print(f"\n📊 RTX 3090 PERFORMANCE RESULTS")
    print("=" * 50)
    
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        
        print(f"✅ Successfully processed: {len(inference_times)} inferences")
        print(f"⚡ Average inference: {avg_time:.4f}s ({avg_time*1000:.1f}ms)")
        print(f"🚀 Fastest: {min_time:.4f}s ({min_time*1000:.1f}ms)")
        print(f"🐌 Slowest: {max_time:.4f}s ({max_time*1000:.1f}ms)")
        
        # Performance grading
        if avg_time < 0.02:
            grade = "A+"
            status = "🚀 BLAZING FAST - RTX 3090 fully optimized!"
        elif avg_time < 0.05:
            grade = "A"
            status = "⚡ VERY FAST - Excellent GPU acceleration"
        elif avg_time < 0.1:
            grade = "B"
            status = "✅ FAST - Good GPU performance"
        elif avg_time < 0.3:
            grade = "C"
            status = "⚠️  MODERATE - Some GPU usage"
        else:
            grade = "D"
            status = "🐌 SLOW - Likely CPU fallback"
            
        print(f"🏆 Performance Grade: {grade}")
        print(f"📈 Status: {status}")
        
        # Processing rate calculations
        images_per_second = 1 / avg_time
        images_per_minute = images_per_second * 60
        images_per_hour = images_per_second * 3600
        
        print(f"\n⚡ Processing Rates:")
        print(f"  📊 {images_per_second:.1f} images/second")
        print(f"  📊 {images_per_minute:.0f} images/minute")
        print(f"  📊 {images_per_hour:.0f} images/hour")
        
        # Estimate for full dataset
        total_images = 6559
        estimated_seconds = total_images / images_per_second
        estimated_minutes = estimated_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        print(f"\n🎯 Full Dataset Processing Time ({total_images:,} images):")
        if estimated_hours < 1:
            print(f"  ⏱️  {estimated_minutes:.1f} minutes")
        elif estimated_hours < 24:
            print(f"  ⏱️  {estimated_hours:.1f} hours")
        else:
            print(f"  ⏱️  {estimated_hours/24:.1f} days")
            
        # Readiness assessment
        print(f"\n🚀 FACE PROCESSING READINESS:")
        if avg_time < 0.1:
            print("  ✅ READY for large-scale processing!")
            print("  🚀 RTX 3090 is properly optimized")
            print("  📈 High-speed face recognition pipeline confirmed")
        else:
            print("  ⚠️  Performance could be better")
            print("  💡 Consider GPU optimization")
    else:
        print("❌ No successful inferences")
        
    # Final GPU status
    print(f"\n🖥️  CURRENT RTX 3090 STATUS:")
    current_gpu = get_gpu_stats()
    if current_gpu:
        parts = current_gpu.split(',')
        if len(parts) >= 6:
            print(f"  🎮 GPU: {parts[0].strip()}")
            print(f"  ⚡ Utilization: {parts[1].strip()}%")
            print(f"  💾 Memory: {parts[2].strip()}%")
            print(f"  📊 Memory Used: {parts[3].strip()}MB / {parts[4].strip()}MB") 
            print(f"  🌡️  Temperature: {parts[5].strip()}°C")
    
    return inference_times

def get_gpu_stats():
    """Get current RTX 3090 stats"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=3)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'RTX 3090' in line:
                    return line
    except:
        pass
    return None

if __name__ == "__main__":
    test_face_inference_in_wsl()
