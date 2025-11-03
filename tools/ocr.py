"""
OCR (Optical Character Recognition) text extraction
Using EasyOCR for robust multi-language support
"""
import cv2
from pathlib import Path
from typing import List, Dict
import easyocr

# Global OCR reader (loaded once)
_reader = None


def initialize(languages=['en']):
    """Initialize EasyOCR reader"""
    global _reader

    if _reader is None:
        print(f"Loading EasyOCR for languages: {languages}")
        _reader = easyocr.Reader(languages, gpu=False)  # CPU mode
        print("EasyOCR loaded successfully")


def extract_text(image_path: str, languages=['en'], detail=True) -> Dict:
    """
    Extract text from an image using OCR

    Args:
        image_path: Path to image file
        languages: List of language codes (e.g., ['en', 'es', 'fr'])
        detail: If True, return bounding boxes and confidence scores

    Returns:
        Dictionary with extracted text and optional details
    """
    initialize(languages)

    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Perform OCR
    results = _reader.readtext(image)

    if not detail:
        # Return just the text
        text = ' '.join([result[1] for result in results])
        return {
            "text": text,
            "words_found": len(results)
        }

    # Return detailed results with bounding boxes
    detailed_results = []
    for bbox, text, confidence in results:
        detailed_results.append({
            "text": text,
            "confidence": float(confidence),
            "bounding_box": {
                "top_left": [float(bbox[0][0]), float(bbox[0][1])],
                "top_right": [float(bbox[1][0]), float(bbox[1][1])],
                "bottom_right": [float(bbox[2][0]), float(bbox[2][1])],
                "bottom_left": [float(bbox[3][0]), float(bbox[3][1])]
            }
        })

    # Combine all text
    full_text = ' '.join([r['text'] for r in detailed_results])

    return {
        "text": full_text,
        "words_found": len(detailed_results),
        "details": detailed_results
    }


def get_status() -> Dict:
    """Get status of OCR system"""
    try:
        initialize()
        return {
            "available": True,
            "engine": "EasyOCR",
            "supported_languages": ['en', 'es', 'fr', 'de', 'it', 'pt', 'zh', 'ja', 'ko']  # subset
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }
