"""
Open WebUI Vision Tool v2
Integrates local vision analysis (Google Coral + Intel NCS2) into OpenWebUI chats
Enhanced with better file handling and debugging
"""

import base64
import json
import requests
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        VISION_SERVER_URL: str = Field(
            default="http://10.0.1.23:8000",
            description="URL of the vision tool server"
        )
        OBJECT_DETECTION_THRESHOLD: float = Field(
            default=0.4,
            description="Confidence threshold for object detection (0.0-1.0)"
        )
        FACE_DETECTION_THRESHOLD: float = Field(
            default=0.5,
            description="Confidence threshold for face detection (0.0-1.0)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.file_handler = True  # Tell OpenWebUI to pass files via __files__

    def _get_file_path(self, __files__: Optional[List[Any]]) -> Optional[str]:
        """Extract file path from OpenWebUI __files__ parameter"""
        if not __files__ or len(__files__) == 0:
            return None

        file_info = __files__[0]

        # Handle different OpenWebUI versions and formats
        if isinstance(file_info, str):
            # Simple string path
            return file_info
        elif isinstance(file_info, dict):
            # Dictionary with various possible keys
            return (
                file_info.get('path') or
                file_info.get('file', {}).get('path') if isinstance(file_info.get('file'), dict) else None or
                file_info.get('url') or
                file_info.get('filepath') or
                file_info.get('file_path') or
                file_info.get('id')  # Sometimes OpenWebUI uses file ID
            )
        elif hasattr(file_info, 'path'):
            # Object with path attribute
            return file_info.path

        return None

    def _encode_image_from_path(self, file_path: str) -> str:
        """Read image file and encode as base64"""
        with open(file_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode('utf-8')

    def _call_vision_api(self, endpoint: str, image_path: str, **kwargs) -> dict:
        """Call the vision server API"""
        try:
            # Encode image to base64
            image_base64 = self._encode_image_from_path(image_path)

            # Prepare request
            url = f"{self.valves.VISION_SERVER_URL}/{endpoint}"
            payload = {
                "image_base64": image_base64,
                **kwargs
            }

            # Make request
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            return {
                "success": False,
                "error": f"Vision API error: {str(e)}"
            }

    def detect_objects(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Detect objects in an image using Google Coral TPU.

        Upload an image and this tool will identify objects, their locations, and confidence scores.
        Returns a list of detected objects with bounding boxes.

        :param __files__: List of uploaded files (images)
        :return: Detected objects and their details
        """
        # Extract file path
        file_path = self._get_file_path(__files__)

        if not file_path:
            # Debug info
            files_debug = f"Files received: {type(__files__)}, count: {len(__files__) if __files__ else 0}"
            if __files__ and len(__files__) > 0:
                files_debug += f", first item type: {type(__files__[0])}"
                if isinstance(__files__[0], dict):
                    files_debug += f", keys: {list(__files__[0].keys())}"
            return f"Error: No image file path found. {files_debug}\n\nPlease upload an image before calling this tool."

        # Call vision API
        result = self._call_vision_api(
            "detect_objects",
            file_path,
            threshold=self.valves.OBJECT_DETECTION_THRESHOLD
        )

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # Format response
        objects = result.get('objects', [])
        count = result.get('count', 0)

        if count == 0:
            return "No objects detected in the image."

        response = f"Detected {count} object(s):\n\n"
        for i, obj in enumerate(objects, 1):
            label = obj.get('label', 'unknown')
            confidence = obj.get('confidence', 0) * 100
            bbox = obj.get('bbox', {})
            response += f"{i}. {label} ({confidence:.1f}% confidence)\n"
            if bbox:
                response += f"   Location: x={bbox.get('xmin', 0):.0f}, y={bbox.get('ymin', 0):.0f}, "
                response += f"width={bbox.get('width', 0):.0f}, height={bbox.get('height', 0):.0f}\n"

        return response

    def classify_image(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
        top_k: int = 5,
    ) -> str:
        """
        Classify an image using Google Coral TPU.

        Upload an image and this tool will identify what it contains from 1000+ categories.
        Returns the top classification predictions with confidence scores.

        :param __files__: List of uploaded files (images)
        :param top_k: Number of top predictions to return (default: 5)
        :return: Top classification predictions
        """
        file_path = self._get_file_path(__files__)

        if not file_path:
            return "Error: No image provided. Please upload an image before calling this tool."

        # Call vision API
        result = self._call_vision_api(
            "classify_image",
            file_path,
            top_k=top_k
        )

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # Format response
        predictions = result.get('predictions', [])

        if not predictions:
            return "No classification predictions available."

        response = f"Top {len(predictions)} classification(s):\n\n"
        for i, pred in enumerate(predictions, 1):
            label = pred.get('label', 'unknown')
            confidence = pred.get('confidence', 0) * 100
            response += f"{i}. {label} ({confidence:.1f}% confidence)\n"

        return response

    def extract_text(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
        languages: str = "en",
    ) -> str:
        """
        Extract text from an image using OCR.

        Upload an image containing text and this tool will extract all readable text.
        Supports multiple languages (use comma-separated language codes like 'en,es,fr').

        :param __files__: List of uploaded files (images)
        :param languages: Comma-separated language codes (default: 'en')
        :return: Extracted text
        """
        file_path = self._get_file_path(__files__)

        if not file_path:
            return "Error: No image provided. Please upload an image before calling this tool."

        # Call vision API
        result = self._call_vision_api(
            "extract_text",
            file_path,
            languages=languages,
            detail=True
        )

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # Format response
        full_text = result.get('full_text', '')
        details = result.get('details', [])

        if not full_text:
            return "No text detected in the image."

        response = f"Extracted Text:\n\n{full_text}\n"

        if details:
            response += f"\n\nDetected {len(details)} text region(s) with position information."

        return response

    def detect_faces(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Detect faces in an image using Intel NCS2.

        Upload an image and this tool will identify faces and their locations.
        Returns the number of faces detected with bounding box information.

        :param __files__: List of uploaded files (images)
        :return: Detected faces and their locations
        """
        file_path = self._get_file_path(__files__)

        if not file_path:
            return "Error: No image provided. Please upload an image before calling this tool."

        # Call vision API
        result = self._call_vision_api(
            "detect_faces",
            file_path,
            threshold=self.valves.FACE_DETECTION_THRESHOLD
        )

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # Format response
        faces = result.get('faces', [])
        count = result.get('count', 0)

        if count == 0:
            return "No faces detected in the image."

        response = f"Detected {count} face(s):\n\n"
        for i, face in enumerate(faces, 1):
            confidence = face.get('confidence', 0) * 100
            bbox = face.get('bbox', {})
            response += f"{i}. Face ({confidence:.1f}% confidence)\n"
            if bbox:
                response += f"   Location: x={bbox.get('xmin', 0):.0f}, y={bbox.get('ymin', 0):.0f}, "
                response += f"width={bbox.get('width', 0):.0f}, height={bbox.get('height', 0):.0f}\n"

        return response

    def test_file_upload(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
    ) -> str:
        """
        Test function to debug file upload issues.

        This tool will tell you exactly what it receives from OpenWebUI.
        Call this first to diagnose file upload problems.

        :param __files__: Files from OpenWebUI
        :return: Debug information
        """
        import sys

        debug_info = []
        debug_info.append(f"Python version: {sys.version}")
        debug_info.append(f"file_handler flag: {getattr(self, 'file_handler', 'NOT SET')}")
        debug_info.append(f"\n__files__ type: {type(__files__)}")
        debug_info.append(f"__files__ value: {__files__}")

        if __files__:
            debug_info.append(f"__files__ length: {len(__files__)}")
            if len(__files__) > 0:
                debug_info.append(f"First item type: {type(__files__[0])}")
                debug_info.append(f"First item value: {__files__[0]}")
                if isinstance(__files__[0], dict):
                    debug_info.append(f"First item keys: {list(__files__[0].keys())}")

        debug_info.append(f"\n__user__ type: {type(__user__)}")
        debug_info.append(f"__user__ keys: {list(__user__.keys()) if __user__ else 'None'}")

        # Try to extract file path
        file_path = self._get_file_path(__files__)
        debug_info.append(f"\nExtracted file_path: {file_path}")

        return "=== FILE UPLOAD DEBUG INFO ===\n\n" + "\n".join(debug_info)

    def analyze_scene(
        self,
        __user__: Optional[Dict] = None,
        __files__: Optional[List[Any]] = None,
        __event_emitter__=None,
        include_text: bool = True,
        include_faces: bool = True,
    ) -> str:
        """
        Perform comprehensive scene analysis on an image.

        Upload an image and this tool will provide a complete analysis including:
        - Object detection (what objects are present)
        - Image classification (what the scene is)
        - OCR text extraction (any text in the image)
        - Face detection (any faces present)

        :param __files__: List of uploaded files (images)
        :param include_text: Include OCR text extraction (default: True)
        :param include_faces: Include face detection (default: True)
        :return: Comprehensive scene analysis
        """
        file_path = self._get_file_path(__files__)

        if not file_path:
            # DETAILED DEBUG INFO
            import sys
            debug_lines = []
            debug_lines.append("=== DIAGNOSTIC INFO ===\n")
            debug_lines.append(f"Python: {sys.version.split()[0]}")
            debug_lines.append(f"file_handler: {getattr(self, 'file_handler', 'NOT SET')}")
            debug_lines.append(f"\n__files__ received:")
            debug_lines.append(f"  Type: {type(__files__)}")
            debug_lines.append(f"  Value: {repr(__files__)}")

            if __files__:
                debug_lines.append(f"  Length: {len(__files__)}")
                if len(__files__) > 0:
                    debug_lines.append(f"  First item type: {type(__files__[0])}")
                    debug_lines.append(f"  First item: {repr(__files__[0])[:500]}")
                    if isinstance(__files__[0], dict):
                        debug_lines.append(f"  Keys: {list(__files__[0].keys())}")

            if __user__:
                debug_lines.append(f"\n__user__ keys: {list(__user__.keys())}")

            debug_lines.append(f"\nExtracted path: {file_path}")
            debug_lines.append("\n" + "="*40)
            debug_lines.append("\nTROUBLESHOOTING:")
            debug_lines.append("1. Make sure image is uploaded BEFORE sending message")
            debug_lines.append("2. Image should be visible in chat")
            debug_lines.append("3. Check if file_handler = True")
            debug_lines.append("4. Try a different model (llama3.1, GPT-4)")

            return "\n".join(debug_lines)

        # Call vision API
        result = self._call_vision_api(
            "analyze_scene",
            file_path,
            include_text=include_text,
            include_faces=include_faces
        )

        if not result.get('success'):
            return f"Error: {result.get('error', 'Unknown error')}"

        # Format response
        summary = result.get('summary', '')
        analysis = result.get('analysis', {})

        if summary:
            return f"Scene Analysis:\n\n{summary}"

        # Fallback to detailed breakdown if summary not available
        response = "Scene Analysis:\n\n"

        # Classification
        classification = analysis.get('classification', {})
        if classification.get('predictions'):
            response += "Main Classifications:\n"
            for pred in classification['predictions'][:3]:
                label = pred.get('label', 'unknown')
                conf = pred.get('confidence', 0) * 100
                response += f"  - {label} ({conf:.1f}%)\n"
            response += "\n"

        # Objects
        objects = analysis.get('objects', {})
        if objects.get('detected'):
            response += f"Detected {len(objects['detected'])} object(s):\n"
            for obj in objects['detected'][:5]:  # Top 5
                label = obj.get('label', 'unknown')
                conf = obj.get('confidence', 0) * 100
                response += f"  - {label} ({conf:.1f}%)\n"
            response += "\n"

        # Text
        if include_text:
            text_data = analysis.get('text', {})
            if text_data.get('full_text'):
                response += f"Text Found: {text_data['full_text']}\n\n"

        # Faces
        if include_faces:
            faces = analysis.get('faces', {})
            if faces.get('count', 0) > 0:
                response += f"Faces Detected: {faces['count']}\n"

        return response
