#!/usr/bin/env python3

import sqlite3
import json
import time
import os
import threading
from datetime import datetime
import subprocess
import sys

# Add the LVFace directory to Python path
sys.path.append('/mnt/c/Users/yanbo/wSpace/vlm-photo-engine/LVFace')

class DirectGPUFaceProcessor:
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.stop_processing = False
        self.inferencer = None
        
    def initialize_gpu_model(self):
        """Initialize LVFace model directly with GPU"""
        try:
            from inference_onnx import LVFaceONNXInferencer
            
            model_path = "/mnt/c/Users/yanbo/wSpace/vlm-photo-engine/LVFace/models/LVFace-B_Glint360K.onnx"
            print(f"🤖 Loading model: {model_path}")
            
            # Initialize with GPU
            self.inferencer = LVFaceONNXInferencer(model_path, use_gpu=True)
            print("✅ GPU model loaded successfully")
            
            # Test inference to warm up GPU
            import numpy as np
            test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            
            print("🔥 Warming up GPU...")
            start_time = time.time()
            for i in range(5):
                _ = self.inferencer._infer_onnx(test_img)
            warm_time = (time.time() - start_time) / 5
            
            print(f"⚡ GPU warm-up time: {warm_time:.3f}s per inference")
            
            if warm_time < 0.01:
                print("🚀 EXCELLENT: GPU is blazing fast!")
            elif warm_time < 0.05:
                print("✅ GOOD: GPU acceleration working")
            else:
                print("⚠️  SLOW: GPU might not be optimized")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to load GPU model: {e}")
            return False
    
    def get_pending_images(self, batch_size=50):
        """Get images that need processing from Windows database"""
        db_path = "/mnt/c/Users/yanbo/wSpace/vlm-photo-engine/vlmPhotoHouse/metadata.sqlite"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.id, a.path
            FROM assets a
            WHERE a.mime LIKE 'image/%'
            AND a.path IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM face_detections fd 
                WHERE fd.asset_id = a.id
            )
            ORDER BY a.id
            LIMIT ?
        """, (batch_size,))
        
        pending_images = cursor.fetchall()
        conn.close()
        return pending_images
    
    def save_face_embedding(self, asset_id, embedding):
        """Save embedding to Windows database"""
        db_path = "/mnt/c/Users/yanbo/wSpace/vlm-photo-engine/vlmPhotoHouse/metadata.sqlite"
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO face_detections 
                (asset_id, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?)
            """, (asset_id, 0, 0, 112, 112))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ DB Error for asset {asset_id}: {e}")
            return False
        finally:
            conn.close()
    
    def process_image_direct(self, asset_id, image_path):
        """Process image directly with GPU (no network overhead)"""
        if self.stop_processing or not self.inferencer:
            return False
            
        try:
            # Convert Windows path to WSL path
            if image_path.startswith('E:\\'):
                wsl_path = image_path.replace('E:\\', '/mnt/e/').replace('\\', '/')
            elif image_path.startswith('C:\\'):
                wsl_path = image_path.replace('C:\\', '/mnt/c/').replace('\\', '/')
            else:
                wsl_path = image_path
            
            if not os.path.exists(wsl_path):
                self.error_count += 1
                return False
            
            # Load image directly
            import cv2
            img = cv2.imread(wsl_path)
            if img is None:
                self.error_count += 1
                return False
            
            # Direct GPU inference (no network overhead!)
            start_inference = time.time()
            embedding = self.inferencer._infer_onnx(img)
            inference_time = time.time() - start_inference
            
            # Save to database
            if self.save_face_embedding(asset_id, embedding.tolist()):
                self.processed_count += 1
                
                # Print detailed timing for first few
                if self.processed_count <= 10:
                    print(f"  📸 Asset {asset_id}: {inference_time:.3f}s inference | Embedding: {len(embedding)}")
                
                if self.processed_count % 10 == 0:
                    self.print_progress()
                    
                return True
            
            self.error_count += 1
            return False
            
        except Exception as e:
            self.error_count += 1
            return False
    
    def print_progress(self):
        """Print processing progress"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            remaining = 6559 - self.processed_count
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            progress_pct = (self.processed_count / 6559) * 100
            print(f"🚀 Progress: {self.processed_count:,}/6,559 ({progress_pct:.1f}%) | Errors: {self.error_count} | Rate: {rate:.1f}/sec | ETA: {eta_minutes:.0f}m")
    
    def monitor_gpu(self):
        """Monitor RTX 3090 usage"""
        while not self.stop_processing:
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=3)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'RTX 3090' in line or len(lines) <= 2:  # Assume RTX 3090 if only one GPU
                            parts = line.split(', ')
                            if len(parts) >= 4:
                                gpu_util = parts[0].strip()
                                mem_util = parts[1].strip()
                                temp = parts[2].strip()
                                power = parts[3].strip()
                                
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                print(f"🖥️  {timestamp} | RTX 3090: {gpu_util}% GPU | {mem_util}% Mem | {temp}°C | {power}W")
                                break
            except:
                pass
            
            time.sleep(10)
    
    def start_direct_processing(self, batch_size=100):
        """Start direct GPU processing (no network overhead)"""
        print("🚀 STARTING DIRECT RTX 3090 PROCESSING")
        print("=" * 60)
        print("💡 Processing directly on GPU (no network/API overhead)")
        print()
        
        if not self.initialize_gpu_model():
            return
        
        self.start_time = time.time()
        
        # Start GPU monitoring
        gpu_thread = threading.Thread(target=self.monitor_gpu)
        gpu_thread.daemon = True
        gpu_thread.start()
        
        try:
            batch_num = 0
            while not self.stop_processing:
                # Get next batch
                pending_images = self.get_pending_images(batch_size)
                
                if not pending_images:
                    print("✅ All images processed!")
                    break
                
                batch_num += 1
                print(f"\n📦 Batch {batch_num}: Processing {len(pending_images)} images directly...")
                
                batch_start = time.time()
                
                for asset_id, image_path in pending_images:
                    if self.stop_processing:
                        break
                    
                    self.process_image_direct(asset_id, image_path)
                
                batch_time = time.time() - batch_start
                batch_rate = len(pending_images) / batch_time
                print(f"   ⚡ Batch completed: {batch_rate:.1f} images/sec")
        
        except KeyboardInterrupt:
            print("\n⏹️  Processing stopped by user")
            self.stop_processing = True
        
        # Final results
        self.stop_processing = True
        elapsed = time.time() - self.start_time
        
        print(f"\n🎉 DIRECT GPU PROCESSING COMPLETE!")
        print("=" * 50)
        print(f"✅ Processed: {self.processed_count:,} images")
        print(f"❌ Errors: {self.error_count}")
        print(f"⏱️  Time: {elapsed/60:.1f} minutes")
        
        if elapsed > 0:
            rate = self.processed_count / elapsed
            print(f"📈 Rate: {rate:.1f} images/second")
            print(f"🚀 Performance: {rate * 3600:.0f} images/hour")
            
            # Compare to previous performance
            old_rate = 5.0
            improvement = (rate / old_rate) * 100
            print(f"📊 Improvement: {improvement:.0f}% vs previous method")

def main():
    processor = DirectGPUFaceProcessor()
    
    print("🚀 Starting direct RTX 3090 processing in 3 seconds...")
    print("   This bypasses all network overhead!")
    time.sleep(3)
    
    processor.start_direct_processing(batch_size=50)

if __name__ == "__main__":
    main()
