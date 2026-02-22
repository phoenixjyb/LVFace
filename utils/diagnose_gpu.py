#!/usr/bin/env python3

import onnxruntime as ort
import numpy as np

print("🔍 DIAGNOSING RTX 3090 + ONNX ISSUE")
print("=" * 50)

# Check available providers
print("Available ONNX providers:", ort.get_available_providers())

# Check if CUDA is available
cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
print(f"CUDA Provider Available: {cuda_available}")

if cuda_available:
    print("\n🚀 Testing CUDA session...")
    try:
        # Create session with CUDA first
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession("./models/LVFace-B_Glint360K.onnx", providers=providers)
        
        actual_providers = session.get_providers()
        print(f"Session is using: {actual_providers}")
        
        if actual_providers[0] == "CUDAExecutionProvider":
            print("✅ GPU DETECTED: Session is using CUDA!")
            
            # Test inference speed
            import time
            dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            # Warm up
            for _ in range(3):
                _ = session.run(None, {input_name: dummy_input})
            
            # Time it
            start = time.time()
            for _ in range(10):
                _ = session.run(None, {input_name: dummy_input})
            end = time.time()
            
            avg_time = (end - start) / 10
            print(f"🚀 GPU Inference time: {avg_time:.4f}s per image")
            print(f"📈 GPU Processing rate: {1/avg_time:.1f} images/second")
            
            if avg_time < 0.01:
                print("🚀 EXCELLENT: True GPU acceleration!")
            elif avg_time < 0.05:
                print("✅ GOOD: GPU is working")
            else:
                print("⚠️ SLOW: GPU might not be fully utilized")
                
        else:
            print("❌ PROBLEM: Session fell back to CPU!")
            print(f"   Using: {actual_providers[0]}")
            
    except Exception as e:
        print(f"❌ Error creating CUDA session: {e}")
        
else:
    print("❌ CUDA Provider not available!")
    print("💡 Check ONNX Runtime GPU installation")

# Check GPU memory and status
print(f"\n🖥️ GPU Status:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'RTX 3090' in line:
                parts = line.split(', ')
                if len(parts) >= 5:
                    print(f"  GPU: {parts[0]}")
                    print(f"  Memory: {parts[2]}MB / {parts[1]}MB")
                    print(f"  Utilization: {parts[3]}%")
                    print(f"  Temperature: {parts[4]}°C")
except:
    print("  ❌ Could not get GPU stats")

print(f"\n💡 RECOMMENDATIONS:")
print("If GPU utilization is low:")
print("  1. Check ONNX Runtime GPU installation")
print("  2. Verify CUDA version compatibility")
print("  3. Check GPU memory allocation")
print("  4. Test with smaller models first")
