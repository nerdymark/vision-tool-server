"""
Image annotation utilities for drawing bounding boxes and labels
"""
import cv2
import base64
import numpy as np
from typing import List, Dict, Optional, Tuple


def annotate_detections(
    image_path: str,
    detections: List[Dict],
    detection_type: str = "object"
) -> str:
    """
    Draw bounding boxes and labels on image and return as base64 string

    Args:
        image_path: Path to the original image
        detections: List of detection dictionaries with bbox and label info
        detection_type: Type of detection ("object", "face", "text")

    Returns:
        Base64-encoded PNG image with annotations
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Define colors for different types (BGR format)
    colors = {
        "object": (0, 255, 0),      # Green
        "face": (255, 0, 0),        # Blue
        "text": (0, 165, 255),      # Orange
        "classification": (147, 20, 255)  # Purple
    }

    color = colors.get(detection_type, (0, 255, 0))

    # Draw each detection
    for det in detections:
        bbox = det.get('bounding_box', {})

        # Handle different bbox formats
        if 'xmin' in bbox:  # Format: {xmin, ymin, xmax, ymax}
            x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
            x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
        elif 'top_left' in bbox:  # Format: {top_left, bottom_right}
            x1, y1 = int(bbox['top_left'][0]), int(bbox['top_left'][1])
            x2, y2 = int(bbox['bottom_right'][0]), int(bbox['bottom_right'][1])
        else:
            continue  # Skip if bbox format unknown

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label_parts = []
        if 'label' in det:
            label_parts.append(det['label'])
        if 'text' in det:
            label_parts.append(det['text'])
        if 'confidence' in det:
            conf = det['confidence']
            label_parts.append(f"{conf*100:.1f}%")

        label = " ".join(label_parts)

        # Draw label background
        if label:
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1  # Filled
            )
            # Draw label text
            cv2.putText(
                img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1
            )

    # Encode to PNG and then to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64


def annotate_scene(
    image_path: str,
    objects: Optional[List[Dict]] = None,
    faces: Optional[List[Dict]] = None,
    text_regions: Optional[List[Dict]] = None
) -> str:
    """
    Draw multiple types of annotations on a single image

    Args:
        image_path: Path to the original image
        objects: List of object detections
        faces: List of face detections
        text_regions: List of text detections (OCR results)

    Returns:
        Base64-encoded PNG image with all annotations
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Define colors (BGR format)
    object_color = (0, 255, 0)      # Green
    face_color = (255, 0, 0)        # Blue
    text_color = (0, 165, 255)      # Orange

    # Draw objects
    if objects:
        for obj in objects:
            _draw_detection(img, obj, object_color, "object")

    # Draw faces
    if faces:
        for face in faces:
            _draw_detection(img, face, face_color, "face")

    # Draw text regions
    if text_regions:
        for text_det in text_regions:
            _draw_detection(img, text_det, text_color, "text")

    # Encode to PNG and then to base64
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64


def _draw_detection(img: np.ndarray, det: Dict, color: Tuple[int, int, int], det_type: str):
    """Helper function to draw a single detection"""
    bbox = det.get('bounding_box', {})

    # Handle different bbox formats
    if 'xmin' in bbox:
        x1, y1 = int(bbox['xmin']), int(bbox['ymin'])
        x2, y2 = int(bbox['xmax']), int(bbox['ymax'])
    elif 'top_left' in bbox:
        x1, y1 = int(bbox['top_left'][0]), int(bbox['top_left'][1])
        x2, y2 = int(bbox['bottom_right'][0]), int(bbox['bottom_right'][1])
    else:
        return

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Prepare label
    label_parts = []
    if 'label' in det:
        label_parts.append(det['label'])
    if 'text' in det:
        # Truncate long text
        text = det['text']
        if len(text) > 20:
            text = text[:17] + "..."
        label_parts.append(text)
    if 'confidence' in det:
        conf = det['confidence']
        label_parts.append(f"{conf*100:.1f}%")

    label = " ".join(label_parts)

    # Draw label
    if label:
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
