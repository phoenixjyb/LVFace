#!/usr/bin/env python3
"""
LVFace GPU-Enabled Demo
=======================

Enhanced demo with GPU support and automatic fallback to CPU when needed.
"""

import argparse
import sys
import time
import os
from pathlib import Path

def setup_cuda_environment():
    """Set up CUDA library paths for GPU inference"""
    venv_path = Path(sys.executable).parent.parent
    nvidia_path = venv_path / "Lib" / "site-packages" / "nvidia"
    
    if nvidia_path.exists():
        cuda_bin_paths = [str(p) for p in nvidia_path.glob("*/bin") if p.is_dir()]
        if cuda_bin_paths:
            current_path = os.environ.get("PATH", "")
            new_path = os.pathsep.join(cuda_bin_paths + [current_path])
            os.environ["PATH"] = new_path
            return True
    return False

def create_smart_inferencer(model_path, prefer_gpu=True):
    """Create inferencer with smart GPU/CPU selection"""
    from inference_onnx import LVFaceONNXInferencer
    
    if prefer_gpu and setup_cuda_environment():
        try:
            print("🚀 Attempting GPU acceleration...")
            inferencer = LVFaceONNXInferencer(model_path, use_gpu=True)
            
            # Test with small dummy input to verify GPU works
            import numpy as np
            test_input = np.random.randn(112, 112, 3).astype(np.uint8)
            _ = inferencer._preprocess_image(test_input)
            
            print("✅ GPU acceleration enabled!")
            return inferencer, "GPU"
            
        except Exception as e:
            print(f"⚠️  GPU failed ({str(e)[:100]}...), falling back to CPU")
    
    print("🔄 Using CPU inference...")
    inferencer = LVFaceONNXInferencer(model_path, use_gpu=False)
    return inferencer, "CPU"

def benchmark_inference(inferencer, device, num_runs=5):
    """Benchmark inference performance"""
    import numpy as np
    import cv2
    
    print(f"\\n📊 Benchmarking {device} inference ({num_runs} runs)...")
    
    # Create test image
    test_img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    cv2.imwrite("temp_benchmark.jpg", test_img)
    
    times = []
    try:
        # Warm up
        _ = inferencer.infer_from_image("temp_benchmark.jpg")
        
        # Benchmark
        for i in range(num_runs):
            start = time.time()
            features = inferencer.infer_from_image("temp_benchmark.jpg")
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
        
        avg_time = sum(times) / len(times)
        print(f"\\n⚡ Average {device} inference time: {avg_time*1000:.2f}ms")
        print(f"🔢 Feature shape: {features.shape}")
        
    finally:
        # Clean up
        if Path("temp_benchmark.jpg").exists():
            Path("temp_benchmark.jpg").unlink()
    
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='LVFace GPU-Enabled Demo')
    parser.add_argument('--model', default='./models/LVFace-B_Glint360K.onnx', 
                       help='Path to ONNX model file')
    parser.add_argument('--img1', help='Path to first image')
    parser.add_argument('--img2', help='Path to second image')
    parser.add_argument('--url', help='URL of image to process')
    parser.add_argument('--cpu-only', action='store_true', 
                       help='Force CPU inference only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return 1
    
    # Create inferencer with smart device selection
    print(f"🚀 Initializing LVFace with {Path(args.model).name}")
    try:
        inferencer, device = create_smart_inferencer(args.model, not args.cpu_only)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    
    # Run benchmark if requested
    if args.benchmark:
        benchmark_inference(inferencer, device)
    
    # Process images
    if args.img1 and args.img2:
        print(f"\\n📸 Comparing two images using {device}...")
        try:
            start_time = time.time()
            feat1 = inferencer.infer_from_image(args.img1)
            feat2 = inferencer.infer_from_image(args.img2)
            similarity = inferencer.calculate_similarity(feat1, feat2)
            total_time = time.time() - start_time
            
            print(f"Image 1: {args.img1}")
            print(f"Image 2: {args.img2}")
            print(f"🎯 Similarity: {similarity:.6f}")
            print(f"⏱️  Total time: {total_time*1000:.2f}ms")
            
            if similarity > 0.7:
                print("📝 Result: Likely the same person ✅")
            elif similarity > 0.5:
                print("📝 Result: Possibly the same person ⚠️")
            else:
                print("📝 Result: Likely different people ❌")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
            
    elif args.img1:
        print(f"\\n📸 Processing single image using {device}...")
        try:
            start_time = time.time()
            features = inferencer.infer_from_image(args.img1)
            total_time = time.time() - start_time
            
            print(f"✅ Features extracted from {args.img1}")
            print(f"📊 Shape: {features.shape}")
            print(f"⏱️  Time: {total_time*1000:.2f}ms")
            print(f"🔢 Sample: {features[0][:5]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
            
    elif args.url:
        print(f"\\n🌐 Processing URL using {device}...")
        try:
            start_time = time.time()
            features = inferencer.infer_from_url(args.url)
            total_time = time.time() - start_time
            
            print(f"✅ Features extracted from URL")
            print(f"📊 Shape: {features.shape}")
            print(f"⏱️  Time: {total_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        print("\\n📋 Usage examples:")
        print("  python demo_gpu.py --benchmark")
        print("  python demo_gpu.py --img1 face.jpg")
        print("  python demo_gpu.py --img1 face1.jpg --img2 face2.jpg")
        print("  python demo_gpu.py --url https://example.com/face.jpg")
        print("  python demo_gpu.py --cpu-only --benchmark")
    
    print(f"\\n🎉 Demo completed using {device}!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
