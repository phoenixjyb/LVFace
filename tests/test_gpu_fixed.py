#!/usr/bin/env python3

print("🔥 TESTING RTX 3090 + ONNX RUNTIME GPU")
print("=" * 50)

try:
    import onnxruntime as ort
    print("✅ onnxruntime imported successfully")
    
    # Check version
    print(f"📦 ONNX Runtime version: {ort.__version__}")
    
    # Check available providers
    providers = ort.get_available_providers()
    print(f"🔧 Available providers: {providers}")
    
    if "CUDAExecutionProvider" in providers:
        print("🚀 CUDA PROVIDER AVAILABLE!")
        
        # Test GPU session creation
        gpu_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession("./models/LVFace-B_Glint360K.onnx", providers=gpu_providers)
        
        actual_providers = session.get_providers()
        print(f"🎯 Session using: {actual_providers}")
        
        if actual_providers[0] == "CUDAExecutionProvider":
            print("🚀 SUCCESS: GPU SESSION CREATED!")
            
            # Test inference speed
            import numpy as np
            import time
            
            # Create dummy input
            dummy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
            input_name = session.get_inputs()[0].name
            
            print("\n⚡ Testing inference speed...")
            
            # Warm up GPU
            for _ in range(5):
                _ = session.run(None, {input_name: dummy_input})
            
            # Time inference
            times = []
            for i in range(10):
                start = time.time()
                result = session.run(None, {input_name: dummy_input})
                end = time.time()
                times.append(end - start)
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            
            print(f"🚀 Average GPU inference: {avg_time:.4f}s ({avg_time*1000:.1f}ms)")
            print(f"⚡ Fastest inference: {min_time:.4f}s ({min_time*1000:.1f}ms)")
            print(f"📈 Processing rate: {1/avg_time:.1f} images/second")
            
            # Performance assessment
            if avg_time < 0.01:
                print("🔥 BLAZING FAST: RTX 3090 at full power!")
                grade = "A+"
            elif avg_time < 0.02:
                print("🚀 VERY FAST: Excellent GPU performance!")
                grade = "A"
            elif avg_time < 0.05:
                print("✅ FAST: Good GPU acceleration")
                grade = "B"
            else:
                print("⚠️ SLOW: GPU might not be fully utilized")
                grade = "C"
                
            print(f"🏆 Performance Grade: {grade}")
            
            # Estimate processing time for full dataset
            total_images = 6559
            estimated_time = total_images * avg_time
            if estimated_time < 60:
                print(f"🎯 Full dataset ETA: {estimated_time:.0f} seconds")
            elif estimated_time < 3600:
                print(f"🎯 Full dataset ETA: {estimated_time/60:.1f} minutes")
            else:
                print(f"🎯 Full dataset ETA: {estimated_time/3600:.1f} hours")
                
        else:
            print("❌ FAILED: Session fell back to CPU")
            print(f"   Actual provider: {actual_providers[0]}")
    else:
        print("❌ CUDA PROVIDER NOT AVAILABLE")
        
except Exception as e:
    print(f"❌ Error: {e}")

# Check GPU status
print(f"\n🖥️ RTX 3090 Status:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'RTX 3090' in line:
                parts = line.split(', ')
                if len(parts) >= 5:
                    print(f"  🎮 GPU: {parts[0]}")
                    print(f"  ⚡ Utilization: {parts[1]}%")
                    print(f"  💾 Memory: {parts[2]}MB / {parts[3]}MB")
                    print(f"  🌡️ Temperature: {parts[4]}°C")
except:
    print("  ❌ Could not get GPU stats")

print(f"\n🚀 Ready to restart LVFace service with RTX 3090 power!")
