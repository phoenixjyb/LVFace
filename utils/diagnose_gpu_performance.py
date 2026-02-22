#!/usr/bin/env python3

import onnxruntime
import time
import numpy as np

print("🔍 DIAGNOSING RTX 3090 GPU UTILIZATION ISSUE")
print("=" * 60)

# Check available providers
print("📊 Available ONNX providers:")
providers = onnxruntime.get_available_providers()
for i, provider in enumerate(providers):
    print(f"  {i+1}. {provider}")

# Check if CUDA is available
cuda_available = 'CUDAExecutionProvider' in providers
print(f"\n🖥️  CUDA Available: {cuda_available}")

# Try to load the model and check its configuration
model_path = "./models/LVFace-B_Glint360K.onnx"

try:
    print(f"\n🤖 Loading model: {model_path}")
    
    # Create session with explicit CUDA provider
    if cuda_available:
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    else:
        session = onnxruntime.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
    
    # Check which provider is actually being used
    print("✅ Model loaded successfully")
    print(f"🔧 Active providers: {session.get_providers()}")
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print(f"📥 Input: {input_info.name} - {input_info.shape} - {input_info.type}")
    print(f"📤 Output: {output_info.name} - {output_info.shape} - {output_info.type}")
    
    # Test inference speed
    print(f"\n⚡ Testing inference speed...")
    
    # Create dummy input data
    input_shape = input_info.shape
    if input_shape[0] == 'batch_size' or input_shape[0] is None:
        input_shape = [1] + list(input_shape[1:])
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warm up
    for _ in range(3):
        _ = session.run([output_info.name], {input_info.name: dummy_input})
    
    # Time multiple inferences
    times = []
    for i in range(10):
        start_time = time.time()
        output = session.run([output_info.name], {input_info.name: dummy_input})
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"  Test {i+1}: {inference_time:.4f}s")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"  ⚡ Average: {avg_time:.4f}s ({avg_time*1000:.1f}ms)")
    print(f"  🚀 Fastest: {min_time:.4f}s ({min_time*1000:.1f}ms)")
    print(f"  🐌 Slowest: {max_time:.4f}s ({max_time*1000:.1f}ms)")
    
    # Performance assessment
    if avg_time < 0.01:
        print("🚀 EXCELLENT: Proper GPU acceleration!")
    elif avg_time < 0.05:
        print("✅ GOOD: Decent GPU performance")
    elif avg_time < 0.1:
        print("⚠️  MODERATE: Some GPU usage")
    else:
        print("🐌 SLOW: Likely CPU fallback!")
    
    # Calculate expected throughput
    images_per_sec = 1 / avg_time
    print(f"📈 Expected throughput: {images_per_sec:.1f} images/second")
    print(f"📊 Expected throughput: {images_per_sec * 3600:.0f} images/hour")
    
    # Check if this matches our observed performance
    observed_rate = 5.0  # from the logs
    efficiency = observed_rate / images_per_sec * 100
    print(f"\n🔍 ANALYSIS:")
    print(f"  Expected rate: {images_per_sec:.1f} images/sec")
    print(f"  Observed rate: {observed_rate:.1f} images/sec")
    print(f"  Efficiency: {efficiency:.1f}%")
    
    if efficiency < 50:
        print("❌ MAJOR BOTTLENECK: GPU not being utilized properly!")
        print("💡 Possible causes:")
        print("   - Image preprocessing overhead")
        print("   - Network/base64 encoding overhead")
        print("   - Database writing overhead")
        print("   - Threading/batching issues")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Also check PyTorch CUDA
try:
    import torch
    print(f"\n🔥 PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("⚠️  PyTorch not available")
