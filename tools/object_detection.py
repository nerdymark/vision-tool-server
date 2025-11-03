"""
Object detection using Google Coral USB Accelerator
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from pycoral.adapters import common, detect
from pycoral.utils.edgetpu import make_interpreter

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models" / "coral"
MODEL_FILE = MODELS_DIR / "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_FILE = MODELS_DIR / "coco_labels.txt"

# Global interpreter (loaded once)
_interpreter = None
_labels = None


def load_labels(path):
    """Load labels from text file"""
    with open(path, 'r') as f:
        lines = f.readlines()
    labels = {}
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # Check if line has "ID label" format or just "label"
        pair = line.split(maxsplit=1)
        if len(pair) == 2 and pair[0].isdigit():
            # Format: "0 background"
            labels[int(pair[0])] = pair[1].strip()
        else:
            # Format: just "person" - use line index as ID
            labels[idx] = line
    return labels


def initialize():
    """Initialize the Coral TPU interpreter"""
    global _interpreter, _labels

    if _interpreter is None:
        print(f"Loading object detection model from {MODEL_FILE}")
        _interpreter = make_interpreter(str(MODEL_FILE))
        _interpreter.allocate_tensors()
        print("Object detection model loaded successfully")

    if _labels is None:
        _labels = load_labels(LABELS_FILE)
        print(f"Loaded {len(_labels)} object labels")


def detect_objects(image_path: str, threshold: float = 0.4) -> List[Dict]:
    """
    Detect objects in an image using Google Coral

    Args:
        image_path: Path to image file
        threshold: Confidence threshold (0.0-1.0)

    Returns:
        List of detected objects with bounding boxes and confidence scores
    """
    initialize()

    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get input size
    _, height, width, _ = _interpreter.get_input_details()[0]['shape']

    # Resize image to model input size
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))

    # Run inference
    common.set_input(_interpreter, image_resized)
    _interpreter.invoke()

    # Get results
    objects = detect.get_objects(_interpreter, threshold)

    # Format results
    results = []
    for obj in objects:
        result = {
            "label": _labels.get(obj.id, obj.id),
            "confidence": float(obj.score),
            "bounding_box": {
                "ymin": float(obj.bbox.ymin),
                "xmin": float(obj.bbox.xmin),
                "ymax": float(obj.bbox.ymax),
                "xmax": float(obj.bbox.xmax)
            }
        }
        results.append(result)

    return results


def get_status() -> Dict:
    """Get status of object detection system"""
    try:
        initialize()
        return {
            "available": True,
            "model": MODEL_FILE.name,
            "device": "Google Coral USB Accelerator",
            "labels_count": len(_labels) if _labels else 0
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
