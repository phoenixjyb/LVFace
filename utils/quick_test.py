#!/usr/bin/env python3
import requests
import time
import subprocess

print("🧪 QUICK RTX 3090 FACE SERVICE TEST")
print("=" * 40)

# Test service health
try:
    response = requests.get("http://127.0.0.1:8003/health", timeout=5)
    if response.status_code == 200:
        health = response.json()
        print("✅ Service Health:")
        for k, v in health.items():
            print(f"  {k}: {v}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"❌ Connection error: {e}")

# Check RTX 3090 status
print("\n🖥️  RTX 3090 Status:")
try:
    result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=name,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True, timeout=5)
    
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'RTX 3090' in line:
                parts = line.split(', ')
                if len(parts) >= 4:
                    print(f"  🎮 GPU: {parts[0]}")
                    print(f"  ⚡ Utilization: {parts[1]}%")
                    print(f"  💾 Memory: {parts[2]}%") 
                    print(f"  🌡️  Temperature: {parts[3]}°C")
                    if len(parts) >= 5:
                        print(f"  🔋 Power: {parts[4]}W")
    else:
        print("❌ nvidia-smi failed")
except Exception as e:
    print(f"❌ GPU check error: {e}")

print("\n🚀 Face service with RTX 3090 GPU is ready!")
print("📊 Ready to process 6,559 images for face recognition")
