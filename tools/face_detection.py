"""
Face detection using Intel NCS2 with OpenVINO
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from openvino.runtime import Core

# Model will be downloaded from OpenVINO Model Zoo
MODEL_NAME = "face-detection-retail-0004"
_ie = None
_compiled_model = None
_input_layer = None
_output_layer = None


def initialize():
    """Initialize OpenVINO and load face detection model"""
    global _ie, _compiled_model, _input_layer, _output_layer

    if _compiled_model is None:
        print("Initializing OpenVINO for face detection...")
        _ie = Core()

        print(f"Loading model: {MODEL_NAME}")

        # Point to downloaded model (FP16 is optimized for NCS2)
        models_dir = Path(__file__).parent.parent / "models" / "intel"
        model_path = models_dir / MODEL_NAME / "FP16"

        try:
            # Load the downloaded model
            xml_path = model_path / f"{MODEL_NAME}.xml"
            bin_path = model_path / f"{MODEL_NAME}.bin"

            if not xml_path.exists():
                print(f"Model not found at {xml_path}")
                print(f"Please download: omz_downloader --name {MODEL_NAME} --output_dir {models_dir}")
                raise FileNotFoundError(f"Model {MODEL_NAME} not found at {xml_path}")

            print(f"Loading model from {xml_path}")
            model = _ie.read_model(model=xml_path)
            _compiled_model = _ie.compile_model(model=model, device_name="MYRIAD")  # MYRIAD = NCS2

            _input_layer = _compiled_model.input(0)
            _output_layer = _compiled_model.output(0)

            print("Face detection model loaded on Intel NCS2")
        except Exception as e:
            print(f"Warning: Could not load on NCS2, falling back to CPU: {e}")
            model = _ie.read_model(model=xml_path) if xml_path.exists() else None
            if model:
                _compiled_model = _ie.compile_model(model=model, device_name="CPU")
                _input_layer = _compiled_model.input(0)
                _output_layer = _compiled_model.output(0)


def detect_faces(image_path: str, threshold: float = 0.5) -> List[Dict]:
    """
    Detect faces in an image using Intel NCS2

    Args:
        image_path: Path to image file
        threshold: Confidence threshold (0.0-1.0)

    Returns:
        List of detected faces with bounding boxes and confidence scores
    """
    try:
        initialize()
    except Exception as e:
        return {
            "error": f"Model not initialized. Please download model first: {str(e)}",
            "instructions": f"Run: omz_downloader --name {MODEL_NAME}"
        }

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get input shape
    n, c, h, w = _input_layer.shape

    # Preprocess
    resized_image = cv2.resize(image, (w, h))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

    # Run inference
    result = _compiled_model([input_image])[_output_layer]

    # Parse results
    faces = []
    image_h, image_w = image.shape[:2]

    for detection in result[0][0]:
        confidence = float(detection[2])
        if confidence > threshold:
            xmin = int(detection[3] * image_w)
            ymin = int(detection[4] * image_h)
            xmax = int(detection[5] * image_w)
            ymax = int(detection[6] * image_h)

            faces.append({
                "confidence": confidence,
                "bounding_box": {
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                }
            })

    return faces


def get_status() -> Dict:
    """Get status of face detection system"""
    try:
        initialize()
        return {
            "available": True,
            "model": MODEL_NAME,
            "device": "Intel NCS2 (MYRIAD)",
            "framework": "OpenVINO"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "instructions": f"Download model: omz_downloader --name {MODEL_NAME}"
        }
