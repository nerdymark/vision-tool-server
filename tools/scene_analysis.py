"""
Comprehensive scene analysis combining multiple AI tools
"""
from typing import Dict
from pathlib import Path

from .object_detection import detect_objects
from .classification import classify_image
from .ocr import extract_text
from .face_detection import detect_faces


def analyze_scene(image_path: str, include_text: bool = True, include_faces: bool = True) -> Dict:
    """
    Perform comprehensive scene analysis on an image

    Args:
        image_path: Path to image file
        include_text: Whether to perform OCR
        include_faces: Whether to detect faces

    Returns:
        Dictionary with comprehensive scene analysis
    """
    results = {
        "image": str(image_path),
        "analysis": {}
    }

    # Object detection (Coral)
    try:
        objects = detect_objects(image_path, threshold=0.3)
        results["analysis"]["objects"] = {
            "count": len(objects),
            "detected": objects
        }
    except Exception as e:
        results["analysis"]["objects"] = {"error": str(e)}

    # Image classification (Coral)
    try:
        classifications = classify_image(image_path, top_k=3)
        results["analysis"]["classification"] = {
            "top_predictions": classifications
        }
    except Exception as e:
        results["analysis"]["classification"] = {"error": str(e)}

    # OCR text extraction
    if include_text:
        try:
            text_data = extract_text(image_path, detail=False)
            results["analysis"]["text"] = text_data
        except Exception as e:
            results["analysis"]["text"] = {"error": str(e)}

    # Face detection (Intel NCS2)
    if include_faces:
        try:
            faces = detect_faces(image_path, threshold=0.5)
            if isinstance(faces, dict) and "error" in faces:
                results["analysis"]["faces"] = faces
            else:
                results["analysis"]["faces"] = {
                    "count": len(faces),
                    "detected": faces
                }
        except Exception as e:
            results["analysis"]["faces"] = {"error": str(e)}

    # Generate human-readable summary
    results["summary"] = generate_summary(results["analysis"])

    return results


def generate_summary(analysis: Dict) -> str:
    """Generate a human-readable summary of the scene"""
    parts = []

    # Classification
    if "classification" in analysis and "top_predictions" in analysis["classification"]:
        top_class = analysis["classification"]["top_predictions"][0]
        parts.append(f"This appears to be {top_class['label']} ({top_class['confidence']:.1%} confident)")

    # Objects
    if "objects" in analysis and "detected" in analysis["objects"]:
        obj_count = analysis["objects"]["count"]
        if obj_count > 0:
            # Count unique objects
            obj_labels = [obj["label"] for obj in analysis["objects"]["detected"]]
            unique_objects = list(set(obj_labels))
            if len(unique_objects) <= 3:
                parts.append(f"I can see: {', '.join(unique_objects)}")
            else:
                parts.append(f"I can see {obj_count} objects including: {', '.join(unique_objects[:3])}, and more")

    # Faces
    if "faces" in analysis and "count" in analysis["faces"]:
        face_count = analysis["faces"]["count"]
        if face_count > 0:
            parts.append(f"{face_count} {'face' if face_count == 1 else 'faces'} detected")

    # Text
    if "text" in analysis and "text" in analysis["text"]:
        text_content = analysis["text"]["text"].strip()
        if text_content:
            word_count = analysis["text"].get("words_found", 0)
            if len(text_content) > 100:
                parts.append(f"Contains text ({word_count} words): {text_content[:100]}...")
            else:
                parts.append(f"Contains text: {text_content}")

    return ". ".join(parts) if parts else "Unable to analyze scene"
