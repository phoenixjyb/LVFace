# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import requests
import cv2
import numpy as np
import onnxruntime
from typing import Optional, Tuple, Union
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

class LVFaceONNXInferencer:
    """LVFace Inference Class using ONNX Runtime"""
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the LVFace ONNX inferencer
        
        Args:
            model_path (str): Path to the ONNX model file
            use_gpu (bool): Whether to use GPU acceleration (requires onnxruntime-gpu)
        """
        # Select execution provider
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Initialize ONNX Runtime session
        self.ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers
        )
        
        # Get input and output names
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        
        # Input image size
        self.input_size = (112, 112)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for LVFace inference
        
        Args:
            img (np.ndarray): Input image in BGR format (from cv2.imread)
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        # Resize image to input size
        img_resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Transpose to (C, H, W)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))
        
        # Normalize to [-1, 1]
        img_normalized = ((img_transposed / 255.0) - 0.5) / 0.5
        
        # Convert to float32 and add batch dimension
        img_tensor = img_normalized.astype(np.float32)[np.newaxis, ...]
        
        return img_tensor

    def infer_from_image(self, img_path: str) -> np.ndarray:
        """
        Extract feature from a local image file
        
        Args:
            img_path (str): Path to the local image file
            
        Returns:
            np.ndarray: Extracted feature embedding
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image from {img_path}")
            
        # Preprocess image
        img_tensor = self._preprocess_image(img)
        
        # Run inference
        output = self.ort_session.run(
            [self.output_name],
            {self.input_name: img_tensor}
        )
        
        return output[0]

    def infer_from_url(self, img_url: str) -> np.ndarray:
        """
        Extract feature from an image URL
        
        Args:
            img_url (str): URL of the image
            
        Returns:
            np.ndarray: Extracted feature embedding
        """
        try:
            # Download image from URL
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Convert to numpy array
            np_arr = np.frombuffer(response.content, np.uint8)
            
            # Decode image
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from URL content")
                
            # Preprocess and infer
            img_tensor = self._preprocess_image(img)
            output = self.ort_session.run(
                [self.output_name],
                {self.input_name: img_tensor}
            )
            
            return output[0]
            
        except Exception as e:
            raise RuntimeError(f"Error processing image from URL: {str(e)}")

    @staticmethod
    def calculate_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two features
        
        Args:
            feat1 (np.ndarray): First feature embedding
            feat2 (np.ndarray): Second feature embedding
            
        Returns:
            float: Cosine similarity score (range: [-1, 1])
        """
        # Flatten features
        feat1_flat = np.ravel(feat1)
        feat2_flat = np.ravel(feat2)
        
        # Calculate cosine similarity
        dot_product = np.dot(feat1_flat, feat2_flat)
        norm1 = np.linalg.norm(feat1_flat)
        norm2 = np.linalg.norm(feat2_flat)
        
        return dot_product / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0


if __name__ == "__main__":
    # Initialize Flask app and model
    app = Flask(__name__)
    model_path = "./models/LVFace-B_Glint360K.onnx"  # Update with your model path
    
    try:
        inferencer = LVFaceONNXInferencer(model_path, use_gpu=True)
        print("✅ LVFace ONNX model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load LVFace model: {e}")
        inferencer = None

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        status = "healthy" if inferencer is not None else "unhealthy"
        return jsonify({
            "status": status,
            "service": "LVFace-ONNX",
            "model_loaded": inferencer is not None
        })

    @app.route('/embed', methods=['POST'])
    def get_face_embedding():
        """Get face embedding from image"""
        if inferencer is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        try:
            # Check if image is provided in base64 format
            if 'image' in request.json:
                # Decode base64 image
                image_data = base64.b64decode(request.json['image'])
                img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            elif 'url' in request.json:
                # Download image from URL
                response = requests.get(request.json['url'])
                img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                return jsonify({"error": "No image or url provided"}), 400
                
            # Get embedding
            embedding = inferencer._infer_onnx(img)
            
            return jsonify({
                "embedding": embedding.tolist(),
                "shape": embedding.shape
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/similarity', methods=['POST'])
    def calculate_similarity():
        """Calculate similarity between two face embeddings"""
        if inferencer is None:
            return jsonify({"error": "Model not loaded"}), 500
            
        try:
            data = request.json
            if 'embedding1' not in data or 'embedding2' not in data:
                return jsonify({"error": "Two embeddings required"}), 400
                
            emb1 = np.array(data['embedding1'])
            emb2 = np.array(data['embedding2'])
            
            similarity = inferencer.calculate_similarity(emb1, emb2)
            
            return jsonify({
                "similarity": float(similarity)
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Start Flask server
    print("🚀 Starting LVFace ONNX service on port 8003...")
    app.run(host='0.0.0.0', port=8003, debug=False)
    
    # Example usage (won't run when used as web server)
    """
    # Example 1: Inference from local image
    try:
        img_path = "1.jpg"  # Update with your image path
        embedding = inferencer.infer_from_image(img_path)
        print(f"Extracted feature shape: {embedding.shape}")
    except Exception as e:
        print(f"Error in image inference: {e}")
    
    # Example 2: Inference from URL
    try:
        img_url = "url"
        embedding_url = inferencer.infer_from_url(img_url)
        print(f"Extracted feature from URL shape: {embedding_url.shape}")
    except Exception as e:
        print(f"Error in URL inference: {e}")
    
    # Example 3: Calculate similarity between two images
    try:
        feat1 = inferencer.infer_from_image(img_path)
        feat2 = inferencer.infer_from_url(img_url)
        similarity = inferencer.calculate_similarity(feat1, feat2)
        print(f"Cosine similarity: {similarity:.6f}")
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
    """