"""
Vision Tool Server - FastAPI server for OpenWebUI
Provides local AI vision capabilities using Google Coral and Intel NCS2
"""
import os
import base64
import tempfile
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our vision tools
from tools import object_detection, classification, ocr, face_detection, scene_analysis

# Import image optimization utilities
from utils import resize_with_retry, get_image_info

# Initialize FastAPI app
app = FastAPI(
    title="Vision Tool Server",
    description="Local AI-powered vision tools using Google Coral and Intel NCS2",
    version="1.0.0"
)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# Request/Response models
class DetectObjectsRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    threshold: float = 0.4

class ClassifyImageRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    top_k: int = 5

class ExtractTextRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    languages: List[str] = ['en']
    detail: bool = True

class DetectFacesRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    threshold: float = 0.5

class AnalyzeSceneRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    include_text: bool = True
    include_faces: bool = True


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
        # Decode base64
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        image_data = base64.b64decode(base64_data)
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


@app.post("/detect_objects")
async def detect_objects_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    threshold: float = Form(0.4)
):
    """
    Detect objects in an image using Google Coral

    Returns list of detected objects with bounding boxes and confidence scores
    """
    try:
        if file:
            img_path, metadata = save_image(file=file)
        elif image_path:
            img_path = Path(image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        results = object_detection.detect_objects(str(img_path), threshold=threshold)
        return {
            "success": True,
            "objects": results,
            "count": len(results),
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/classify_image")
async def classify_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    top_k: int = Form(5)
):
    """
    Classify an image using Google Coral

    Returns top K predictions with labels and confidence scores
    """
    try:
        if file:
            img_path, metadata = save_image(file=file)
        elif image_path:
            img_path = Path(image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        results = classification.classify_image(str(img_path), top_k=top_k)
        return {
            "success": True,
            "predictions": results,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/extract_text")
async def extract_text_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    languages: str = Form("en"),
    detail: bool = Form(True)
):
    """
    Extract text from an image using OCR

    Returns extracted text with optional bounding boxes and confidence scores
    """
    try:
        if file:
            img_path, metadata = save_image(file=file)
        elif image_path:
            img_path = Path(image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        lang_list = languages.split(',')
        results = ocr.extract_text(str(img_path), languages=lang_list, detail=detail)
        return {
            "success": True,
            **results,
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/detect_faces")
async def detect_faces_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    threshold: float = Form(0.5)
):
    """
    Detect faces in an image using Intel NCS2

    Returns list of detected faces with bounding boxes and confidence scores
    """
    try:
        if file:
            img_path, metadata = save_image(file=file)
        elif image_path:
            img_path = Path(image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        results = face_detection.detect_faces(str(img_path), threshold=threshold)
        if isinstance(results, dict) and "error" in results:
            return JSONResponse(status_code=500, content={"success": False, **results})

        return {
            "success": True,
            "faces": results,
            "count": len(results),
            "image_metadata": metadata
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/analyze_scene")
async def analyze_scene_endpoint(
    file: Optional[UploadFile] = File(None),
    image_path: Optional[str] = Form(None),
    include_text: bool = Form(True),
    include_faces: bool = Form(True)
):
    """
    Perform comprehensive scene analysis on an image

    Combines object detection, classification, OCR, and face detection
    Returns detailed analysis and human-readable summary
    """
    try:
        if file:
            img_path, metadata = save_image(file=file)
        elif image_path:
            img_path = Path(image_path)
            metadata = {}
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        results = scene_analysis.analyze_scene(
            str(img_path),
            include_text=include_text,
            include_faces=include_faces
        )
        return {
            "success": True,
            **results,
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
