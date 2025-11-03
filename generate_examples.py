#!/usr/bin/env python3
"""
Generate examples for README documentation
Tests all endpoints and saves results to examples/ folder
"""
import os
import json
import base64
import requests
from pathlib import Path

# Configuration
API_BASE = "http://localhost:8000"
EXAMPLES_DIR = Path(__file__).parent / "examples"
EXAMPLES_DIR.mkdir(exist_ok=True)

# Test images - using original test images from uploads folder
TEST_IMAGES = {
    "test_face": "/home/mark/vision-tool-server/uploads/test_face.jpg",
    "test_text": "/home/mark/vision-tool-server/uploads/test_text.png",
    "test_landmark": "/home/mark/vision-tool-server/uploads/test_landmark.jpg",
    "test_vision": "/home/mark/vision-tool-server/uploads/test_vision.jpg",
}

def save_base64_image(base64_str, output_path):
    """Save a base64 encoded image to file"""
    if base64_str:
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(base64_str))
        print(f"  ✓ Saved: {output_path}")
        return True
    return False

def test_object_detection(image_name, image_path):
    """Test object detection endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing Object Detection: {image_name}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'threshold': 0.4}
        response = requests.post(f"{API_BASE}/detect_objects", files=files, data=data)

    if response.status_code == 200:
        result = response.json()

        # Save original image
        output_dir = EXAMPLES_DIR / "object_detection"
        output_dir.mkdir(exist_ok=True)
        os.system(f"cp {image_path} {output_dir}/{image_name}_original.jpg")

        # Save annotated image
        if result.get('annotated_image'):
            save_base64_image(
                result['annotated_image'],
                output_dir / f"{image_name}_annotated.png"
            )

        # Save JSON response (without base64 image for readability)
        result_copy = result.copy()
        if 'annotated_image' in result_copy and result_copy['annotated_image']:
            result_copy['annotated_image'] = f"<base64 image data - {len(result['annotated_image'])} chars>"

        with open(output_dir / f"{image_name}_result.json", 'w') as f:
            json.dump(result_copy, f, indent=2)

        print(f"\n  Objects detected: {result.get('count', 0)}")
        for obj in result.get('objects', [])[:3]:
            print(f"    - {obj['label']}: {obj['confidence']*100:.1f}%")

        return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        return None

def test_classification(image_name, image_path):
    """Test image classification endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing Classification: {image_name}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'top_k': 5}
        response = requests.post(f"{API_BASE}/classify_image", files=files, data=data)

    if response.status_code == 200:
        result = response.json()

        # Save original image
        output_dir = EXAMPLES_DIR / "classification"
        output_dir.mkdir(exist_ok=True)
        os.system(f"cp {image_path} {output_dir}/{image_name}_original.jpg")

        # Save JSON response
        with open(output_dir / f"{image_name}_result.json", 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n  Top predictions:")
        for pred in result.get('predictions', [])[:3]:
            print(f"    - {pred['label']}: {pred['confidence']*100:.1f}%")

        return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        return None

def test_face_detection(image_name, image_path):
    """Test face detection endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing Face Detection: {image_name}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'threshold': 0.5}
        response = requests.post(f"{API_BASE}/detect_faces", files=files, data=data)

    if response.status_code == 200:
        result = response.json()

        # Save original image
        output_dir = EXAMPLES_DIR / "face_detection"
        output_dir.mkdir(exist_ok=True)
        os.system(f"cp {image_path} {output_dir}/{image_name}_original.jpg")

        # Save annotated image
        if result.get('annotated_image'):
            save_base64_image(
                result['annotated_image'],
                output_dir / f"{image_name}_annotated.png"
            )

        # Save JSON response (without base64 image for readability)
        result_copy = result.copy()
        if 'annotated_image' in result_copy and result_copy['annotated_image']:
            result_copy['annotated_image'] = f"<base64 image data - {len(result['annotated_image'])} chars>"

        with open(output_dir / f"{image_name}_result.json", 'w') as f:
            json.dump(result_copy, f, indent=2)

        print(f"\n  Faces detected: {result.get('count', 0)}")

        return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        return None

def test_ocr(image_name, image_path):
    """Test OCR endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing OCR: {image_name}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'languages': 'en', 'detail': True}
        response = requests.post(f"{API_BASE}/extract_text", files=files, data=data)

    if response.status_code == 200:
        result = response.json()

        # Save original image
        output_dir = EXAMPLES_DIR / "ocr"
        output_dir.mkdir(exist_ok=True)
        os.system(f"cp {image_path} {output_dir}/{image_name}_original.jpg")

        # Save annotated image
        if result.get('annotated_image'):
            save_base64_image(
                result['annotated_image'],
                output_dir / f"{image_name}_annotated.png"
            )

        # Save JSON response (without base64 image for readability)
        result_copy = result.copy()
        if 'annotated_image' in result_copy and result_copy['annotated_image']:
            result_copy['annotated_image'] = f"<base64 image data - {len(result['annotated_image'])} chars>"

        with open(output_dir / f"{image_name}_result.json", 'w') as f:
            json.dump(result_copy, f, indent=2)

        print(f"\n  Text extracted: {len(result.get('text', '').strip())} chars")
        if result.get('text'):
            print(f"  Preview: {result['text'][:100]}...")

        return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        return None

def test_scene_analysis(image_name, image_path):
    """Test scene analysis endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing Scene Analysis: {image_name}")
    print(f"{'='*60}")

    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'include_text': True, 'include_faces': True}
        response = requests.post(f"{API_BASE}/analyze_scene", files=files, data=data)

    if response.status_code == 200:
        result = response.json()

        # Save original image
        output_dir = EXAMPLES_DIR / "scene_analysis"
        output_dir.mkdir(exist_ok=True)
        os.system(f"cp {image_path} {output_dir}/{image_name}_original.jpg")

        # Save annotated image
        if result.get('annotated_image'):
            save_base64_image(
                result['annotated_image'],
                output_dir / f"{image_name}_annotated.png"
            )

        # Save JSON response (without base64 image for readability)
        result_copy = result.copy()
        if 'annotated_image' in result_copy and result_copy['annotated_image']:
            result_copy['annotated_image'] = f"<base64 image data - {len(result['annotated_image'])} chars>"

        with open(output_dir / f"{image_name}_result.json", 'w') as f:
            json.dump(result_copy, f, indent=2)

        print(f"\n  Summary: {result.get('summary', 'N/A')}")

        return result
    else:
        print(f"  ✗ Error: {response.status_code}")
        return None

def main():
    print("=" * 60)
    print("Vision Tool Server - Example Generation")
    print("=" * 60)

    # Test each endpoint with each image
    for image_name, image_path in TEST_IMAGES.items():
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        test_object_detection(image_name, image_path)
        test_classification(image_name, image_path)
        test_face_detection(image_name, image_path)
        test_ocr(image_name, image_path)
        test_scene_analysis(image_name, image_path)

    print(f"\n{'='*60}")
    print(f"Examples saved to: {EXAMPLES_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
