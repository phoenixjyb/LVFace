#!/usr/bin/env python3

import requests
import base64
import time
import subprocess
import threading

def monitor_gpu():
    """Monitor GPU during test"""
    for i in range(10):
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3:
                            gpu_util = parts[0].strip()
                            temp = parts[1].strip()
                            power = parts[2].strip()
                            print(f"⚡ GPU: {gpu_util}% | Temp: {temp}°C | Power: {power}W")
                            break
        except:
            pass
        time.sleep(1)

def test_service_gpu_usage():
    """Test if the service actually uses GPU"""
    print("🧪 TESTING LVFace SERVICE GPU USAGE")
    print("=" * 50)
    
    # Create a test image (small dummy image)
    import io
    from PIL import Image
    import numpy as np
    
    # Create a 224x224 test image
    test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format='JPEG')
    img_data = img_buffer.getvalue()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    
    print("📸 Created test image (224x224)")
    
    # Start GPU monitoring
    gpu_thread = threading.Thread(target=monitor_gpu)
    gpu_thread.daemon = True
    gpu_thread.start()
    
    print("🚀 Starting rapid inference test...")
    
    # Send multiple rapid requests to stress test GPU
    times = []
    for i in range(20):
        print(f"🔍 Test {i+1}/20...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://127.0.0.1:8003/embed",
                json={'image': img_b64},
                timeout=10
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                embedding_size = len(result.get('embedding', []))
                times.append(inference_time)
                print(f"  ✅ {inference_time:.3f}s | Embedding: {embedding_size}")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Exception: {str(e)[:50]}...")
        
        # Small delay
        time.sleep(0.1)
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        print(f"\n📊 RESULTS:")
        print(f"✅ Successful inferences: {len(times)}")
        print(f"⚡ Average time: {avg_time:.3f}s ({avg_time*1000:.0f}ms)")
        print(f"🚀 Fastest time: {min_time:.3f}s ({min_time*1000:.0f}ms)")
        print(f"📈 Theoretical max rate: {1/avg_time:.1f} images/sec")
        
        # Assessment
        if avg_time < 0.02:
            print("🚀 EXCELLENT: Proper GPU acceleration!")
        elif avg_time < 0.05:
            print("✅ GOOD: Decent GPU performance")
        elif avg_time < 0.1:
            print("⚠️  MODERATE: Limited GPU usage")
        else:
            print("🐌 SLOW: CPU fallback detected!")
            
        print(f"\n🔍 DIAGNOSIS:")
        current_rate = 5.0  # From your logs
        if avg_time * 4 > 1/current_rate:  # Factor of 4 for overhead
            print("❌ BOTTLENECK: Service overhead is the issue")
            print("💡 Solutions:")
            print("   - Reduce image encoding overhead")
            print("   - Implement batching")
            print("   - Optimize network transfer")
            print("   - Use direct file processing")
        else:
            print("❌ BOTTLENECK: GPU not being used efficiently")
            print("💡 Solutions:")
            print("   - Check ONNX Runtime CUDA configuration")
            print("   - Verify model is loaded on GPU")
            print("   - Check CUDA libraries")

if __name__ == "__main__":
    test_service_gpu_usage()
