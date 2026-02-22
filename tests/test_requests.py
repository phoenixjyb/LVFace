#!/usr/bin/env python3
"""Test requests module functionality"""

try:
    import requests
    print("✅ requests imported successfully")
    
    # Test if requests.get exists
    if hasattr(requests, 'get'):
        print("✅ requests.get method exists")
    else:
        print("❌ requests.get method missing")
        
    # Test basic functionality
    print("🔍 Testing basic requests functionality...")
    response = requests.get("https://httpbin.org/get", timeout=10)
    print(f"✅ HTTP request successful: {response.status_code}")
    
except Exception as e:
    print(f"❌ requests test failed: {e}")
    import traceback
    traceback.print_exc()
