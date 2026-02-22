#!/usr/bin/env python3
"""
LVFace Model Inference Guide
===========================

Comprehensive guide for running LVFace model inference with all available methods.
"""

def show_available_scripts():
    """Show all available inference scripts"""
    print("📦 AVAILABLE INFERENCE SCRIPTS")
    print("=" * 50)
    
    print("\n🎯 MAIN INFERENCE SCRIPTS:")
    print("1. 🚀 demo_gpu.py - Enhanced demo with GPU/CPU smart selection")
    print("2. 📋 inference_onnx.py - Core ONNX inference class")
    print("3. 🎨 demo.py - Basic demo script")
    print("4. ⚡ inference.py - Simple inference wrapper")
    
    print("\n🔧 EVALUATION SCRIPTS:")
    print("5. 📊 eval_ijbc.py - IJB-C benchmark evaluation")
    print("6. 🧪 onnx_ijbc.py - ONNX-specific IJB-C evaluation")

def demo_gpu_usage():
    """Show demo_gpu.py usage examples"""
    print("\n🚀 DEMO_GPU.PY - RECOMMENDED FOR NEW USERS")
    print("=" * 50)
    
    print("\n📋 BASIC USAGE:")
    print("# Run basic demo (downloads model if needed)")
    print("python demo_gpu.py")
    
    print("\n📊 PERFORMANCE BENCHMARK:")
    print("# Test inference speed (CPU/GPU)")
    print("python demo_gpu.py --benchmark")
    
    print("\n🖼️  SINGLE IMAGE INFERENCE:")
    print("# Process single image")
    print("python demo_gpu.py --img1 path/to/face.jpg")
    
    print("\n🔄 FACE COMPARISON:")
    print("# Compare two faces (similarity score)")
    print("python demo_gpu.py --img1 face1.jpg --img2 face2.jpg")
    
    print("\n🌐 URL INFERENCE:")
    print("# Process image from URL")
    print("python demo_gpu.py --url https://example.com/face.jpg")
    
    print("\n💻 FORCE CPU ONLY:")
    print("# Disable GPU acceleration")
    print("python demo_gpu.py --cpu-only --benchmark")
    
    print("\n🎯 CUSTOM MODEL:")
    print("# Use different model file")
    print("python demo_gpu.py --model ./path/to/model.onnx --benchmark")

def inference_onnx_usage():
    """Show inference_onnx.py usage as a library"""
    print("\n📋 INFERENCE_ONNX.PY - PYTHON LIBRARY")
    print("=" * 50)
    
    print("\n💻 PYTHON CODE EXAMPLE:")
    print("""
from inference_onnx import LVFaceONNXInferencer

# Initialize inferencer (GPU by default)
inferencer = LVFaceONNXInferencer("./models/LVFace-B_Glint360K.onnx")

# OR initialize with CPU only
inferencer = LVFaceONNXInferencer("./models/LVFace-B_Glint360K.onnx", use_gpu=False)

# Infer from image file
features = inferencer.infer_from_image("face.jpg")
print(f"Feature shape: {features.shape}")  # (1, 512)

# Infer from URL
features = inferencer.infer_from_url("https://example.com/face.jpg")

# Compare two faces
similarity = inferencer.calculate_similarity("face1.jpg", "face2.jpg")
print(f"Similarity: {similarity:.4f}")

# Process numpy array directly
import cv2
img = cv2.imread("face.jpg")
features = inferencer.infer_from_array(img)
""")

def performance_expectations():
    """Show performance expectations"""
    print("\n⚡ PERFORMANCE EXPECTATIONS")
    print("=" * 50)
    
    print("\n📊 CURRENT SETUP (Quadro P2000):")
    print("• Device: CPU (Pascal GPU limitation)")
    print("• Inference Time: ~55ms per image")
    print("• Throughput: ~18 FPS")
    print("• Memory Usage: ~2GB RAM")
    print("• Feature Size: 512 dimensions")
    
    print("\n🚀 WITH RTX 3090 (Future):")
    print("• Device: CUDA GPU")
    print("• Inference Time: ~5-10ms per image")
    print("• Throughput: ~100-200 FPS")
    print("• Memory Usage: ~1-2GB VRAM")
    print("• Speedup: 5-10x faster")

def practical_examples():
    """Show practical usage examples"""
    print("\n🎯 PRACTICAL EXAMPLES")
    print("=" * 50)
    
    print("\n1. 🧪 QUICK TEST:")
    print("python demo_gpu.py --benchmark")
    print("   → Tests your setup and shows performance")
    
    print("\n2. 📱 FACE VERIFICATION:")
    print("python demo_gpu.py --img1 person1.jpg --img2 person2.jpg")
    print("   → Returns similarity score (0.0-1.0)")
    print("   → >0.5 typically indicates same person")
    
    print("\n3. 🏭 BATCH PROCESSING:")
    print("# Create a batch processing script:")
    print("""
from inference_onnx import LVFaceONNXInferencer
import glob

inferencer = LVFaceONNXInferencer("./models/LVFace-B_Glint360K.onnx")

for img_path in glob.glob("./faces/*.jpg"):
    features = inferencer.infer_from_image(img_path)
    print(f"{img_path}: {features.shape}")
""")
    
    print("\n4. 🔍 FACE SEARCH:")
    print("# Build face database and search:")
    print("""
import numpy as np
from inference_onnx import LVFaceONNXInferencer

inferencer = LVFaceONNXInferencer("./models/LVFace-B_Glint360K.onnx")

# Build database
database = {}
for person_img in ["alice.jpg", "bob.jpg", "charlie.jpg"]:
    features = inferencer.infer_from_image(person_img)
    database[person_img] = features

# Search for query image
query_features = inferencer.infer_from_image("unknown_person.jpg")
for name, db_features in database.items():
    similarity = np.dot(query_features.flatten(), db_features.flatten())
    print(f"Similarity to {name}: {similarity:.4f}")
""")

def model_information():
    """Show model information"""
    print("\n📁 MODEL INFORMATION")
    print("=" * 50)
    
    print("\n🎯 LVFACE-B MODEL:")
    print("• File: LVFace-B_Glint360K.onnx")
    print("• Size: ~90MB")
    print("• Input: 112x112 RGB images")
    print("• Output: 512-dimensional embeddings")
    print("• Accuracy: 90%+ on IJB-C benchmark")
    print("• Training: Glint360K dataset")
    
    print("\n📊 MODEL PERFORMANCE:")
    print("• False Acceptance Rate: <0.1%")
    print("• True Acceptance Rate: >99%")
    print("• Robustness: Good with pose/lighting variations")
    print("• Speed: Optimized for real-time inference")

def troubleshooting():
    """Show troubleshooting tips"""
    print("\n🔧 TROUBLESHOOTING")
    print("=" * 50)
    
    print("\n❓ MODEL NOT FOUND:")
    print("• Download with: python inference_onnx.py")
    print("• Or manually place model in ./models/ directory")
    
    print("\n❓ GPU NOT WORKING:")
    print("• Check: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("• Verify: nvidia-smi shows your GPU")
    print("• Normal: Quadro P2000 falls back to CPU (expected)")
    
    print("\n❓ SLOW PERFORMANCE:")
    print("• CPU inference: ~55ms is normal")
    print("• Check virtual environment is activated")
    print("• Close other applications for better performance")
    
    print("\n❓ IMPORT ERRORS:")
    print("• Install: pip install -r requirements.txt")
    print("• Check: python -c \"import onnxruntime, cv2, numpy\"")

def next_steps():
    """Show next steps"""
    print("\n🚀 NEXT STEPS")
    print("=" * 50)
    
    print("\n🎯 GET STARTED:")
    print("1. Run benchmark: python demo_gpu.py --benchmark")
    print("2. Test with your images: python demo_gpu.py --img1 your_face.jpg")
    print("3. Compare faces: python demo_gpu.py --img1 face1.jpg --img2 face2.jpg")
    
    print("\n📚 LEARN MORE:")
    print("• Study inference_onnx.py for API details")
    print("• Modify demo_gpu.py for your use case")
    print("• Build applications using the inferencer class")
    
    print("\n⚡ OPTIMIZE:")
    print("• Install RTX 3090 for 5-10x speedup")
    print("• Batch process multiple images")
    print("• Integrate into your applications")

if __name__ == "__main__":
    print("🎯 LVFACE MODEL INFERENCE GUIDE")
    print("=" * 60)
    
    show_available_scripts()
    demo_gpu_usage()
    inference_onnx_usage()
    performance_expectations()
    practical_examples()
    model_information()
    troubleshooting()
    next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 Start with: python demo_gpu.py --benchmark")
    print("📚 Full API in: inference_onnx.py")
    print("💡 Questions? Check troubleshooting section above!")
    print("=" * 60)
