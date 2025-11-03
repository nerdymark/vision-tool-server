#!/usr/bin/env python3
"""
Quick system test for Vision Tool Server
Tests each component individually
"""
import sys
from pathlib import Path

print("=" * 60)
print("Vision Tool Server - System Test")
print("=" * 60)

# Test 1: Check devices
print("\n[1] Checking USB devices...")
import subprocess
result = subprocess.run(['lsusb'], capture_output=True, text=True)
if '1a6e:089a' in result.stdout:
    print("✓ Google Coral detected")
else:
    print("✗ Google Coral NOT detected")

if '03e7:2485' in result.stdout:
    print("✓ Intel NCS2 detected")
else:
    print("✗ Intel NCS2 NOT detected")

# Test 2: Check models
print("\n[2] Checking AI models...")
models_dir = Path(__file__).parent / "models"

coral_model = models_dir / "coral" / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
if coral_model.exists():
    print(f"✓ Coral object detection model: {coral_model.name}")
else:
    print(f"✗ Coral model missing")

coral_labels = models_dir / "coral" / "coco_labels.txt"
if coral_labels.exists():
    print(f"✓ COCO labels: {coral_labels.name}")
else:
    print(f"✗ COCO labels missing")

ncs2_model = models_dir / "openvino" / "intel" / "face-detection-retail-0004" / "FP16" / "face-detection-retail-0004.xml"
if ncs2_model.exists():
    print(f"✓ NCS2 face detection model: {ncs2_model.name}")
else:
    print(f"✗ NCS2 model missing")

# Test 3: Try importing tools
print("\n[3] Testing Python modules...")
try:
    from tools import object_detection
    print("✓ object_detection module")
except Exception as e:
    print(f"✗ object_detection: {e}")

try:
    from tools import classification
    print("✓ classification module")
except Exception as e:
    print(f"✗ classification: {e}")

try:
    from tools import ocr
    print("✓ ocr module")
except Exception as e:
    print(f"✗ ocr: {e}")

try:
    from tools import face_detection
    print("✓ face_detection module")
except Exception as e:
    print(f"✗ face_detection: {e}")

try:
    from tools import scene_analysis
    print("✓ scene_analysis module")
except Exception as e:
    print(f"✗ scene_analysis: {e}")

# Test 4: Check tool status
print("\n[4] Checking tool status...")
try:
    status = object_detection.get_status()
    if status.get('available'):
        print(f"✓ Object Detection: {status.get('device')}")
    else:
        print(f"✗ Object Detection: {status.get('error')}")
except Exception as e:
    print(f"✗ Object Detection error: {e}")

try:
    status = classification.get_status()
    if status.get('available'):
        print(f"✓ Classification: {status.get('device')}")
    else:
        print(f"✗ Classification: {status.get('error')}")
except Exception as e:
    print(f"✗ Classification error: {e}")

try:
    status = ocr.get_status()
    if status.get('available'):
        print(f"✓ OCR: {status.get('engine')}")
    else:
        print(f"✗ OCR: {status.get('error')}")
except Exception as e:
    print(f"✗ OCR error: {e}")

try:
    status = face_detection.get_status()
    if status.get('available'):
        print(f"✓ Face Detection: {status.get('device')}")
    else:
        print(f"⚠ Face Detection: {status.get('error', 'Not initialized')}")
except Exception as e:
    print(f"⚠ Face Detection error: {e}")

print("\n" + "=" * 60)
print("Test complete! Check results above.")
print("=" * 60)
