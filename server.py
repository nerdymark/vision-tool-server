"""
Vision Tool Server - FastAPI server for OpenWebUI
Provides local AI vision capabilities using Google Coral and Intel NCS2
"""
import os
import base64
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our vision tools
from tools import object_detection, classification, ocr, face_detection, scene_analysis

# Import image optimization utilities
from utils import resize_with_retry, get_image_info, annotate_detections, annotate_scene

# Initialize FastAPI app
app = FastAPI(
    title="Vision Tool Server",
    description="Local AI-powered vision tools using Google Coral and Intel NCS2",
    version="1.0.0"
)

# Add CORS middleware to allow OpenWebUI to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (OpenWebUI on any port/host)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# Request/Response models
class DetectObjectsRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image file on server")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    threshold: float = Field(0.4, description="Confidence threshold (0.0-1.0)")

class ClassifyImageRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image file on server")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    top_k: int = Field(5, description="Number of top predictions to return")

class ExtractTextRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image file on server")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    languages: str = Field('en', description="Comma-separated language codes (e.g., 'en,es,fr')")
    detail: bool = Field(True, description="Include bounding boxes and confidence scores")

class DetectFacesRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image file on server")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    threshold: float = Field(0.5, description="Confidence threshold (0.0-1.0)")

class AnalyzeSceneRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to image file on server")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    include_text: bool = Field(True, description="Include OCR text extraction")
    include_faces: bool = Field(True, description="Include face detection")


def save_image(file: UploadFile = None, base64_data: str = None,
               optimize: bool = True) -> tuple[Path, dict]:
    """
    Save uploaded or base64 image to temp file with optional optimization

    Args:
        file: Uploaded file
        base64_data: Base64 encoded image data
        optimize: Whether to optimize image for token budget (default True)

    Returns:
        Tuple of (image_path, metadata_dict)
    """
    metadata = {}

    if file:
        filepath = UPLOAD_DIR / file.filename
        with open(filepath, 'wb') as f:
            f.write(file.file.read())
    elif base64_data:
        # Check for OpenWebUI placeholder tokens like [img-0]
        if base64_data.startswith('[img-') and base64_data.endswith(']'):
            raise HTTPException(
                status_code=400,
                detail=f"Image placeholder '{base64_data}' detected. OpenWebUI may not be sending actual image data to tools. This is a known limitation - images might not be passed to external tools yet."
            )

        # Decode base64
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]

        try:
            image_data = base64.b64decode(base64_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")

        # Verify we got actual image data
        if len(image_data) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Image data too small ({len(image_data)} bytes). Possible placeholder or invalid data."
            )

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=UPLOAD_DIR)
        temp_file.write(image_data)
        temp_file.close()
        filepath = Path(temp_file.name)
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    # Optimize image if requested
    if optimize:
        try:
            # Get image info
            info = get_image_info(str(filepath))
            metadata['original_info'] = info

            # If image exceeds token budget, resize with retry
            if not info['within_budget']:
                optimized_path, resize_metadata = resize_with_retry(str(filepath))
                metadata['optimization'] = resize_metadata
                filepath = Path(optimized_path)
                print(f"Image optimized: {resize_metadata.get('token_reduction', 'N/A')} token reduction")
            else:
                metadata['optimization'] = {'status': 'no_resize_needed'}
        except Exception as e:
            # Log warning but continue - don't fail the request
            print(f"Warning: Image optimization failed: {e}")
            metadata['optimization'] = {'error': str(e), 'status': 'failed'}

    return filepath, metadata


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Vision Tool Server",
        "version": "1.0.0",
        "status": "running",
        "devices": {
            "coral": "Google Coral USB Accelerator",
            "ncs2": "Intel Neural Compute Stick 2"
        },
        "endpoints": [
            "/detect_objects",
            "/classify_image",
            "/extract_text",
            "/detect_faces",
            "/analyze_scene",
            "/health"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint - verify all systems"""
    return {
        "status": "healthy",
        "tools": {
            "object_detection": object_detection.get_status(),
            "classification": classification.get_status(),
            "ocr": ocr.get_status(),
            "face_detection": face_detection.get_status()
        }
    }


@app.post("/detect_objects", summary="Detect objects in image")
async def detect_objects_endpoint(
    request: DetectObjectsRequest = Body(...)
):
    """
    Detect objects in an image using Google Coral TPU.

    Provide either image_base64 (base64 encoded image) or image_path (server file path).
    Returns list of detected objects with bounding boxes, labels, and confidence scores.
    """
    try:
        if request.image_base64:
            img_path, metadata = save_image(base64_data=request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        results = object_detection.detect_objects(str(img_path), threshold=request.threshold)

        # Generate annotated image with object bounding boxes
        annotated_image = None
        if results:
            try:
                annotated_image = annotate_detections(str(img_path), results, "object")
            except Exception as e:
                print(f"Warning: Could not generate annotated image: {e}")

        return {
            "success": True,
            "objects": results,
            "count": len(results),
            "annotated_image": annotated_image,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/classify_image", summary="Classify image")
async def classify_image_endpoint(
    request: ClassifyImageRequest = Body(...)
):
    """
    Classify an image using Google Coral TPU.

    Provide either image_base64 (base64 encoded image) or image_path (server file path).
    Returns top K classification predictions with labels and confidence scores.
    """
    try:
        if request.image_base64:
            img_path, metadata = save_image(base64_data=request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        results = classification.classify_image(str(img_path), top_k=request.top_k)
        return {
            "success": True,
            "predictions": results,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/extract_text", summary="Extract text from image (OCR)")
async def extract_text_endpoint(
    request: ExtractTextRequest = Body(...)
):
    """
    Extract text from an image using OCR (Optical Character Recognition).

    Provide either image_base64 (base64 encoded image) or image_path (server file path).
    Returns extracted text with optional bounding boxes and confidence scores.
    Supports multiple languages (comma-separated language codes).
    """
    try:
        if request.image_base64:
            img_path, metadata = save_image(base64_data=request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        lang_list = request.languages.split(',')
        results = ocr.extract_text(str(img_path), languages=lang_list, detail=request.detail)

        # Generate annotated image with text bounding boxes
        annotated_image = None
        if results and results.get('details'):
            try:
                annotated_image = annotate_detections(str(img_path), results['details'], "text")
            except Exception as e:
                print(f"Warning: Could not generate annotated image: {e}")

        return {
            "success": True,
            **results,
            "annotated_image": annotated_image,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/detect_faces", summary="Detect faces in image")
async def detect_faces_endpoint(
    request: DetectFacesRequest = Body(...)
):
    """
    Detect faces in an image using Intel NCS2 (Neural Compute Stick 2).

    Provide either image_base64 (base64 encoded image) or image_path (server file path).
    Returns list of detected faces with bounding boxes and confidence scores.
    """
    try:
        if request.image_base64:
            img_path, metadata = save_image(base64_data=request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        results = face_detection.detect_faces(str(img_path), threshold=request.threshold)
        if isinstance(results, dict) and "error" in results:
            return JSONResponse(status_code=500, content={"success": False, **results})

        # Generate annotated image with face bounding boxes
        annotated_image = None
        if results:
            try:
                annotated_image = annotate_detections(str(img_path), results, "face")
            except Exception as e:
                print(f"Warning: Could not generate annotated image: {e}")

        return {
            "success": True,
            "faces": results,
            "count": len(results),
            "annotated_image": annotated_image,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/analyze_scene", summary="Comprehensive scene analysis")
async def analyze_scene_endpoint(
    request: AnalyzeSceneRequest = Body(...)
):
    """
    Perform comprehensive scene analysis on an image.

    Provide either image_base64 (base64 encoded image) or image_path (server file path).
    Combines object detection, image classification, OCR text extraction, and face detection.
    Returns detailed analysis with human-readable summary.
    """
    try:
        if request.image_base64:
            img_path, metadata = save_image(base64_data=request.image_base64)
        elif request.image_path:
            img_path = Path(request.image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_path must be provided")

        results = scene_analysis.analyze_scene(
            str(img_path),
            include_text=request.include_text,
            include_faces=request.include_faces
        )

        # Generate annotated image with all detections
        annotated_image = None
        try:
            analysis = results.get('analysis', {})
            annotated_image = annotate_scene(
                str(img_path),
                objects=analysis.get('objects', {}).get('detected'),
                faces=analysis.get('faces', {}).get('detected'),
                text_regions=analysis.get('text', {}).get('details') if request.include_text else None
            )
        except Exception as e:
            print(f"Warning: Could not generate annotated image: {e}")

        return {
            "success": True,
            **results,
            "annotated_image": annotated_image,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
