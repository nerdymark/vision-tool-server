"""
Image classification using Google Coral USB Accelerator
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from pycoral.adapters import common, classify
from pycoral.utils.edgetpu import make_interpreter

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models" / "coral"
MODEL_FILE = MODELS_DIR / "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
LABELS_FILE = MODELS_DIR / "imagenet_labels.txt"

# Global interpreter (loaded once)
_interpreter = None
_labels = None


def load_labels(path):
    """Load labels from text file"""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def initialize():
    """Initialize the Coral TPU interpreter"""
    global _interpreter, _labels

    if _interpreter is None:
        print(f"Loading classification model from {MODEL_FILE}")
        _interpreter = make_interpreter(str(MODEL_FILE))
        _interpreter.allocate_tensors()
        print("Classification model loaded successfully")

    if _labels is None:
        _labels = load_labels(LABELS_FILE)
        print(f"Loaded {len(_labels)} classification labels")


def classify_image(image_path: str, top_k: int = 5) -> List[Dict]:
    """
    Classify an image using Google Coral

    Args:
        image_path: Path to image file
        top_k: Number of top predictions to return

    Returns:
        List of top predictions with labels and confidence scores
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
    classes = classify.get_classes(_interpreter, top_k=top_k)

    # Format results
    results = []
    for c in classes:
        result = {
            "label": _labels[c.id] if c.id < len(_labels) else f"Unknown ({c.id})",
            "confidence": float(c.score)
        }
        results.append(result)

    return results


def get_status() -> Dict:
    """Get status of classification system"""
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
