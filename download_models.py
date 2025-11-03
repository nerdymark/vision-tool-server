#!/usr/bin/env python3
"""
Download pre-trained models for Coral and OpenVINO
"""
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
CORAL_DIR = MODELS_DIR / "coral"
OPENVINO_DIR = MODELS_DIR / "openvino"

# Ensure directories exist
CORAL_DIR.mkdir(parents=True, exist_ok=True)
OPENVINO_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, dest):
    """Download a file with progress"""
    print(f"Downloading {dest.name}...")
    if dest.exists():
        print(f"  {dest.name} already exists, skipping")
        return

    urllib.request.urlretrieve(url, dest)
    print(f"  Downloaded to {dest}")

def download_coral_models():
    """Download Google Coral models"""
    print("\n=== Downloading Google Coral Models ===")

    # Object detection model (SSD MobileNet V2 - COCO)
    models = {
        "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite":
            "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        "coco_labels.txt":
            "https://github.com/google-coral/test_data/raw/master/coco_labels.txt",
        # Image classification model (MobileNet V2 - ImageNet)
        "mobilenet_v2_1.0_224_quant_edgetpu.tflite":
            "https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_quant_edgetpu.tflite",
        "imagenet_labels.txt":
            "https://github.com/google-coral/test_data/raw/master/imagenet_labels.txt",
    }

    for filename, url in models.items():
        download_file(url, CORAL_DIR / filename)

def download_openvino_models():
    """Download Intel OpenVINO models"""
    print("\n=== Downloading OpenVINO Models ===")
    print("Note: OpenVINO models will be downloaded on first use via openvino toolkit")
    print("Models will be cached in ~/.cache/openvino/")

    # We'll use the OpenVINO Model Zoo downloader in the tools instead
    # For now, just create a placeholder
    (OPENVINO_DIR / "README.txt").write_text(
        "OpenVINO models will be automatically downloaded on first use.\n"
        "Models are cached in ~/.cache/openvino/\n"
    )

if __name__ == "__main__":
    print("Downloading AI models for Vision Tool Server")
    download_coral_models()
    download_openvino_models()
    print("\n=== Model download complete ===")
    print(f"Coral models: {CORAL_DIR}")
    print(f"OpenVINO models: {OPENVINO_DIR}")
