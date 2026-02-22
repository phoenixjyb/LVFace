#!/usr/bin/env python3

import sqlite3
import requests
import json
import time
import base64
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class FaceProcessingOrchestrator:
    def __init__(self, db_path="metadata.sqlite", service_url="http://127.0.0.1:8003"):
        self.db_path = db_path
        self.service_url = service_url
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        self.lock = threading.Lock()
        self.stop_processing = False
        
    def get_pending_images(self, batch_size=50):
        """Get batch of images that need face processing"""
        conn = sqlite3.connect(self.db_path)
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
    
    def save_face_embedding(self, asset_id, embedding, confidence=0.0):
        """Save face embedding to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save embedding as face detection
            cursor.execute("""
                INSERT INTO face_detections 
                (asset_id, bbox_x, bbox_y, bbox_width, bbox_height, confidence, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                asset_id,
                0, 0, 0, 0,  # Placeholder bbox since we're doing embeddings
                confidence,
                json.dumps(embedding)
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"❌ DB Error for asset {asset_id}: {e}")
            return False
        finally:
            conn.close()
    
    def process_single_image(self, asset_id, image_path):
        """Process a single image for face embedding"""
        if self.stop_processing:
            return False
            
        try:
            # Convert Windows path to WSL path if needed
            if image_path.startswith('E:\\'):
                wsl_path = image_path.replace('E:\\', '/mnt/e/').replace('\\', '/')
            elif image_path.startswith('C:\\'):
                wsl_path = image_path.replace('C:\\', '/mnt/c/').replace('\\', '/')
            else:
                wsl_path = image_path
            
            # Check if file exists
            if not os.path.exists(wsl_path):
                with self.lock:
                    self.error_count += 1
                return False
            
            # Read and encode image
            with open(wsl_path, 'rb') as img_file:
                image_data = img_file.read()
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Call face embedding service
            response = requests.post(
                f"{self.service_url}/embed",
                json={'image': image_b64},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                
                if embedding:
                    # Save embedding to database
                    if self.save_face_embedding(asset_id, embedding):
                        with self.lock:
                            self.processed_count += 1
                            if self.processed_count % 10 == 0:
                                self.print_progress()
                        return True
            
            with self.lock:
                self.error_count += 1
            return False
            
        except Exception as e:
            with self.lock:
                self.error_count += 1
            return False
    
    def print_progress(self):
        """Print current processing progress"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            rate = self.processed_count / elapsed if elapsed > 0 else 0
            eta_seconds = (6559 - self.processed_count) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            if eta_hours > 1:
                eta_str = f"{eta_hours:.1f}h"
            elif eta_minutes > 1:
                eta_str = f"{eta_minutes:.0f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
            
            print(f"🚀 Processed: {self.processed_count:,} | Errors: {self.error_count} | Rate: {rate:.1f}/sec | ETA: {eta_str}")
    
    def monitor_gpu(self):
        """Monitor RTX 3090 GPU usage during processing"""
        import subprocess
        
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
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 4:
                                gpu_util = parts[0].strip()
                                mem_util = parts[1].strip()
                                temp = parts[2].strip()
                                power = parts[3].strip()
                                
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                print(f"🖥️  {timestamp} | GPU: {gpu_util}% | Mem: {mem_util}% | {temp}°C | {power}W")
                                break
                            
            except Exception as e:
                print(f"⚠️  GPU monitoring error: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def start_processing(self, max_workers=4, batch_size=50):
        """Start the face processing pipeline"""
        print("🚀 STARTING LARGE-SCALE FACE PROCESSING")
        print("=" * 60)
        print(f"🎯 Target: 6,559 images with RTX 3090 acceleration")
        print(f"🔧 Workers: {max_workers}")
        print(f"📦 Batch size: {batch_size}")
        print()
        
        self.start_time = time.time()
        
        # Start GPU monitoring in background
        gpu_thread = threading.Thread(target=self.monitor_gpu)
        gpu_thread.daemon = True
        gpu_thread.start()
        
        try:
            while not self.stop_processing:
                # Get next batch of images
                pending_images = self.get_pending_images(batch_size)
                
                if not pending_images:
                    print("✅ All images processed!")
                    break
                
                print(f"\n📦 Processing batch of {len(pending_images)} images...")
                
                # Process batch with thread pool
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for asset_id, image_path in pending_images:
                        if self.stop_processing:
                            break
                            
                        future = executor.submit(
                            self.process_single_image, 
                            asset_id, image_path
                        )
                        futures.append(future)
                    
                    # Wait for batch completion
                    for future in as_completed(futures):
                        if self.stop_processing:
                            break
                        future.result()
                
                # Brief pause between batches
                if not self.stop_processing:
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n⏹️  Processing stopped by user")
            self.stop_processing = True
        
        # Final stats
        self.stop_processing = True
        elapsed = time.time() - self.start_time
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print("=" * 40)
        print(f"✅ Successfully processed: {self.processed_count:,} images")
        print(f"❌ Errors: {self.error_count}")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"📈 Average rate: {self.processed_count/elapsed:.1f} images/second")
        
        return self.processed_count, self.error_count

def main():
    # Check service health first
    print("🧪 Checking LVFace service...")
    try:
        response = requests.get("http://127.0.0.1:8003/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ Service Status:")
            for k, v in health.items():
                print(f"  {k}: {v}")
        else:
            print(f"❌ Service unhealthy: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return
    
    print()
    
    # Create and start orchestrator
    orchestrator = FaceProcessingOrchestrator()
    
    print("🚀 Starting face processing in 3 seconds...")
    print("   Press Ctrl+C to stop gracefully")
    time.sleep(3)
    
    processed, errors = orchestrator.start_processing(max_workers=4, batch_size=25)
    
    print(f"\n📊 Final Results: {processed:,} processed, {errors} errors")

if __name__ == "__main__":
    main()
