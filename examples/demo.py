#!/usr/bin/env python3
"""
LVFace Demo Script
==================

This script demonstrates how to use LVFace for face recognition tasks:
1. Extract face embeddings from images
2. Calculate similarity between faces
3. Batch processing multiple images

Usage:
    python demo.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg
    python demo.py --url https://example.com/face.jpg
"""

import argparse
import sys
from pathlib import Path
from inference_onnx import LVFaceONNXInferencer


def main():
    parser = argparse.ArgumentParser(description='LVFace Demo')
    parser.add_argument('--model', default='./models/LVFace-B_Glint360K.onnx', 
                       help='Path to ONNX model file')
    parser.add_argument('--img1', help='Path to first image')
    parser.add_argument('--img2', help='Path to second image')
    parser.add_argument('--url', help='URL of image to process')
    parser.add_argument('--cpu', action='store_true', 
                       help='Force CPU inference (default: try GPU first)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("Please download the model from HuggingFace first.")
        return 1
    
    # Initialize inferencer
    print(f"🚀 Initializing LVFace with model: {Path(args.model).name}")
    try:
        inferencer = LVFaceONNXInferencer(
            model_path=args.model,
            use_gpu=not args.cpu
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Process images
    if args.img1 and args.img2:
        # Compare two images
        print(f"\n📸 Processing images for comparison...")
        try:
            feat1 = inferencer.infer_from_image(args.img1)
            feat2 = inferencer.infer_from_image(args.img2)
            
            similarity = inferencer.calculate_similarity(feat1, feat2)
            
            print(f"Image 1: {args.img1}")
            print(f"Image 2: {args.img2}")
            print(f"🎯 Similarity Score: {similarity:.6f}")
            
            # Interpretation
            if similarity > 0.7:
                print("📝 Interpretation: Likely the same person")
            elif similarity > 0.5:
                print("📝 Interpretation: Possibly the same person")
            else:
                print("📝 Interpretation: Likely different people")
                
        except Exception as e:
            print(f"❌ Error processing images: {e}")
            return 1
            
    elif args.url:
        # Process single URL
        print(f"\n🌐 Processing image from URL...")
        try:
            features = inferencer.infer_from_url(args.url)
            print(f"✅ Successfully extracted features from URL")
            print(f"📊 Feature shape: {features.shape}")
            print(f"🔢 Feature sample: {features[0][:5]}")
        except Exception as e:
            print(f"❌ Error processing URL: {e}")
            return 1
            
    elif args.img1:
        # Process single image
        print(f"\n📸 Processing single image...")
        try:
            features = inferencer.infer_from_image(args.img1)
            print(f"✅ Successfully extracted features from {args.img1}")
            print(f"📊 Feature shape: {features.shape}")
            print(f"🔢 Feature sample: {features[0][:5]}")
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return 1
    else:
        print("❌ Please provide images to process:")
        print("  --img1 IMAGE_PATH                    (single image)")
        print("  --img1 IMG1 --img2 IMG2              (compare two images)")
        print("  --url IMAGE_URL                      (image from URL)")
        return 1
    
    print("\n🎉 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
