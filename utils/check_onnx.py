#!/usr/bin/env python3

import onnxruntime as ort
print("ONNX Runtime version:", ort.__version__)

# Try different methods to check providers
try:
    providers = ort.get_available_providers()
    print("Available providers:", providers)
except AttributeError:
    print("get_available_providers() not available - old ONNX version")
    
try:
    providers = ort.get_all_providers()
    print("All providers:", providers)
except AttributeError:
    print("get_all_providers() not available")

# Check if we can create a session
try:
    session = ort.InferenceSession("./models/LVFace-B_Glint360K.onnx")
    print("Default session providers:", session.get_providers())
except Exception as e:
    print("Error creating session:", e)
