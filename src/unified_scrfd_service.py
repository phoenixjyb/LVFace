#!/usr/bin/env python3
"""
Unified SCRFD Face Detection + LVFace Recognition Service
Enhanced version of LVFace inference_onnx.py with SCRFD face detection
"""

import cv2
import numpy as np
import onnxruntime as ort
import requests
import sqlite3
import json
import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

# Import InsightFace for SCRFD
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    print("✅ InsightFace imported successfully")
except ImportError as e:
    INSIGHTFACE_AVAILABLE = False
    print(f"⚠️ InsightFace not available: {e}")

class UnifiedFaceService:
    def __init__(self):
        self.app = Flask(__name__)
        
        # Initialize ONNX providers
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 
        
        # Load LVFace recognition model
        self.load_lvface_model()
        
        # Load SCRFD face detector
        self.load_scrfd_detector()
        
        # Database connection
        self.db_path = '/mnt/c/Users/yanbo/wSpace/vlm-photo-engine/vlmPhotoHouse/metadata.sqlite'
        
        self.setup_routes()
        
    def load_lvface_model(self):
        """Load the LVFace ONNX model"""
        model_path = 'models/LVFace-B_Glint360K.onnx'
        if not os.path.exists(model_path):
            print(f"❌ Model file {model_path} not found!")
            return False
            
        try:
            self.session = ort.InferenceSession(model_path, providers=self.providers)
            print(f"✅ LVFace model loaded with providers: {self.session.get_providers()}")
            
            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print(f"📐 Model input: {self.input_name}, shape: {self.input_shape}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load LVFace model: {e}")
            return False
            
    def load_scrfd_detector(self):
        """Load SCRFD face detector from InsightFace"""
        try:
            if not INSIGHTFACE_AVAILABLE:
                print("❌ InsightFace not available, falling back to OpenCV")
                return self.load_opencv_fallback()
                
            # Initialize SCRFD face analysis
            print("🔄 Initializing SCRFD FaceAnalysis...")
            self.face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            
            print("📥 Preparing SCRFD models (downloading if needed)...")
            print("   This may take several minutes for first-time setup...")
            
            # Check if models exist before downloading
            import os
            model_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
            if os.path.exists(model_dir):
                print(f"✅ Found existing models in {model_dir}")
            else:
                print(f"📦 Models will be downloaded to {model_dir}")
                
            # Prepare models with progress indication
            import time
            start_time = time.time()
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            end_time = time.time()
            
            self.detector_type = "scrfd"
            print(f"✅ SCRFD face detector loaded with GPU acceleration")
            print(f"⏱️ Model preparation took {end_time - start_time:.1f} seconds")
            return True
            
        except Exception as e:
            print(f"⚠️ SCRFD failed ({str(e)[:100]}...), falling back to OpenCV")
            return self.load_opencv_fallback()
    
    def load_opencv_fallback(self):
        """Fallback to OpenCV face detector"""
        try:
            # Load OpenCV face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detector_type = "opencv_cascade"
            print("✅ OpenCV face detector loaded (fallback)")
            return True
        except Exception as e:
            print(f"❌ Failed to load OpenCV face detector: {e}")
            return False
    
    def detect_faces(self, image):
        """
        Detect faces in image using SCRFD or OpenCV fallback
        Returns: List of face bounding boxes and additional info
        """
        try:
            if self.detector_type == "scrfd" and hasattr(self, 'face_app'):
                return self.detect_faces_scrfd(image)
            else:
                return self.detect_faces_opencv(image)
                
        except Exception as e:
            print(f"❌ Face detection error: {e}")
            return []
    
    def detect_faces_scrfd(self, image):
        """Detect faces using SCRFD"""
        try:
            # SCRFD expects RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB for SCRFD
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Run SCRFD detection
            faces = self.face_app.get(image_rgb)
            
            # Convert to bounding box format
            face_boxes = []
            for face in faces:
                # SCRFD returns bbox as [x1, y1, x2, y2] 
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                # Debug: print conversion
                print(f"🔍 SCRFD bbox conversion: [{x1}, {y1}, {x2}, {y2}] -> [{x1}, {y1}, {w}, {h}]")
                
                face_info = {
                    'bbox': [int(x1), int(y1), int(w), int(h)],  # Return [x1, y1, w, h]
                    'confidence': float(face.det_score),
                    'landmarks': face.kps.tolist() if hasattr(face, 'kps') else None,
                    'detector': 'scrfd'
                }
                face_boxes.append(face_info)
            
            print(f"🔍 SCRFD detected {len(face_boxes)} faces")
            return face_boxes
            
        except Exception as e:
            print(f"❌ SCRFD detection error: {e}")
            return []
    
    def detect_faces_opencv(self, image):
        """Fallback OpenCV face detection"""
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Convert to standard format
            face_boxes = []
            for (x, y, w, h) in faces:
                face_info = {
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'confidence': 0.95,  # OpenCV doesn't provide confidence
                    'landmarks': None,
                    'detector': 'opencv_cascade'
                }
                face_boxes.append(face_info)
            
            print(f"🔍 OpenCV detected {len(face_boxes)} faces")
            return face_boxes
            
        except Exception as e:
            print(f"❌ OpenCV detection error: {e}")
            return []
    
    def preprocess_face(self, face_img):
        """Preprocess face image for LVFace model"""
        try:
            # Resize to model input size
            target_size = (112, 112)  # Standard face recognition input size
            face_resized = cv2.resize(face_img, target_size)
            
            # Normalize
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Convert BGR to RGB and transpose to CHW
            face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
            face_transposed = np.transpose(face_rgb, (2, 0, 1))
            
            # Add batch dimension
            face_batch = np.expand_dims(face_transposed, axis=0)
            
            return face_batch
            
        except Exception as e:
            print(f"❌ Face preprocessing error: {e}")
            return None
    
    def get_face_embedding(self, face_img):
        """Get face embedding using LVFace"""
        try:
            # Preprocess face
            face_input = self.preprocess_face(face_img)
            if face_input is None:
                return None
                
            # Run inference
            outputs = self.session.run(None, {self.input_name: face_input})
            embedding = outputs[0][0]  # Remove batch dimension
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            print(f"❌ Face embedding error: {e}")
            return None
    
    def save_face_detection(self, image_path, face_detections, embeddings):
        """Save face detection results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for i, (face_info, embedding) in enumerate(zip(face_detections, embeddings)):
                if embedding is None:
                    continue
                    
                bbox = face_info['bbox']
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                confidence = float(face_info.get('confidence', 0.95))
                detector_model = face_info.get('detector', 'unknown')
                
                # Create embedding file path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                embedding_filename = f"face_{timestamp}_{i}.json"
                # Use Drive E location for embeddings
                embeddings_dir = "E:/VLM_DATA/embeddings/faces"
                embedding_path = f"{embeddings_dir}/{embedding_filename}"
                
                # Save embedding to file
                os.makedirs(embeddings_dir, exist_ok=True)
                with open(embedding_path, 'w') as f:
                    # Ensure embedding is JSON serializable
                    serializable_embedding = [float(x) for x in embedding] if embedding else []
                    json.dump(serializable_embedding, f)
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO face_detections 
                    (asset_id, bbox_x, bbox_y, bbox_w, bbox_h, confidence, 
                     embedding_path, detection_model, created_at)
                    VALUES (
                        (SELECT id FROM assets WHERE path = ?),
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    image_path, x, y, w, h, confidence,
                    embedding_path, f"{detector_model}_lvface", datetime.now()
                ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Saved {len(embeddings)} face detections to database")
            return True
            
        except Exception as e:
            print(f"❌ Database save error: {e}")
            return False
    
    def process_image(self, image_path):
        """Process single image: detect faces + get embeddings"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}
            
            # Detect faces
            face_detections = self.detect_faces(image)
            if not face_detections:
                return {"faces": 0, "embeddings": []}
            
            # Get embeddings for each face
            embeddings = []
            for face_info in face_detections:
                bbox = face_info['bbox']
                x, y, w, h = bbox
                face_crop = image[y:y+h, x:x+w]
                embedding = self.get_face_embedding(face_crop)
                embeddings.append(embedding)
            
            # Save to database
            self.save_face_detection(image_path, face_detections, embeddings)
            
            return {
                "faces": len(face_detections),
                "detector": self.detector_type,
                "detections": [
                    {
                        "bbox": face_info['bbox'],
                        "confidence": face_info.get('confidence', 0.0),
                        "detector": face_info.get('detector', 'unknown'),
                        "embedding": emb,  # Include actual embedding
                        "embedding_size": len(emb) if emb else 0,
                        "has_landmarks": face_info.get('landmarks') is not None
                    }
                    for face_info, emb in zip(face_detections, embeddings)
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/status', methods=['GET'])
        def status():
            return jsonify({
                "status": "running",
                "service": "unified_scrfd_lvface",
                "providers": self.session.get_providers() if hasattr(self, 'session') else [],
                "face_detector": getattr(self, 'detector_type', 'unknown'),
                "insightface_available": INSIGHTFACE_AVAILABLE
            })
        
        @self.app.route('/process_image', methods=['POST'])
        def process_image_endpoint():
            data = request.json
            image_path = data.get('image_path')
            
            if not image_path:
                return jsonify({"error": "image_path required"}), 400
                
            result = self.process_image(image_path)
            return jsonify(result)
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({"status": "healthy"})

def main():
    print("🚀 Starting Unified SCRFD + LVFace Service")
    
    # Initialize service
    service = UnifiedFaceService()
    
    # Start Flask server
    print("🌐 Starting server on port 8003...")
    service.app.run(host='0.0.0.0', port=8003, debug=False)

if __name__ == "__main__":
    main()
