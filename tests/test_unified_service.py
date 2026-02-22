#!/usr/bin/env python3
print("Testing unified service...")
try:
    from unified_scrfd_service import UnifiedFaceService
    print("✅ Service imported successfully")
    service = UnifiedFaceService()
    print("✅ Service created successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
