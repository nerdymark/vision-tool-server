# Vision Tool Server - Claude Development Notes

**Project:** Local AI Vision Tools for OpenWebUI
**Location:** `/home/mark/vision-tool-server`
**Created:** November 1, 2025
**Status:** ✅ Operational

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation Journey](#installation-journey)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Maintenance](#maintenance)
6. [Development Notes](#development-notes)

---

## Project Overview

### Purpose
Provides local AI-powered vision analysis tools for OpenWebUI using hardware accelerators (Google Coral TPU and Intel NCS2), enabling:
- Private image analysis without cloud services
- Fast inference using edge AI hardware
- Integration with LLM conversations

### Hardware Requirements
- **Google Coral USB Accelerator** - For object detection and classification
- **Intel Neural Compute Stick 2 (NCS2)** - For face detection
- USB 3.0 ports for optimal performance

### Software Stack
- **Python 3.9** (Required - pycoral only has pre-built wheels for 3.9)
- **FastAPI** - REST API server
- **Uvicorn** - ASGI server
- **pycoral** - Google Coral library (with native extensions)
- **OpenVINO 2022.1.0** - Intel NCS2 inference engine (with custom-built MYRIAD plugin)
- **EasyOCR** - Optical character recognition
- **Tesseract** - Fallback OCR engine

### New Features (November 2, 2025)
- **Image Token Optimization** - Automatic image resizing to prevent LLM token overflow
- **Smart Geometric Scaling** - Mathematical estimation using sqrt(target/current) for optimal quality
- **Exponential Backoff Retry** - Progressive token targets (3500, 2800, 2000 tokens)
- **MYRIAD Plugin Support** - Custom-built OpenVINO 2022.1.0 plugin for NCS2 hardware acceleration

### New Features (November 3, 2025)
- **OpenWebUI Python Tool Integration** - Native Python tool that wraps the vision API
- **File Handler Support** - Receives image file paths directly from OpenWebUI
- **CORS Support** - Cross-origin requests enabled for browser-based access
- **JSON API Endpoints** - All endpoints support JSON with base64 image data

---

## Architecture

### API Endpoints

| Endpoint | Method | Hardware | Description |
|----------|--------|----------|-------------|
| `/` | GET | - | Server info |
| `/health` | GET | - | Health check |
| `/detect_objects` | POST | Coral | Detect objects with bounding boxes |
| `/classify_image` | POST | Coral | Classify image into categories |
| `/extract_text` | POST | CPU | OCR text extraction |
| `/detect_faces` | POST | NCS2 | Face detection with age/gender |
| `/analyze_scene` | POST | All | Combined comprehensive analysis |
| `/docs` | GET | - | Swagger UI documentation |
| `/openapi.json` | GET | - | OpenAPI specification |

### Directory Structure
```
/home/mark/vision-tool-server/
├── server.py                           # Main FastAPI application (with CORS)
├── openwebui_vision_tool.py            # OpenWebUI Python tool (NEW)
├── OPENWEBUI_TOOL_INSTALLATION.md      # Tool installation guide (NEW)
├── utils/                              # Utility modules
│   ├── __init__.py
│   └── image_optimizer.py              # Image token estimation and resizing
├── tools/                              # Vision tool modules
│   ├── __init__.py
│   ├── object_detection.py             # Coral - SSD MobileNet v2
│   ├── classification.py               # Coral - MobileNet v2
│   ├── ocr.py                          # EasyOCR/Tesseract
│   ├── face_detection.py               # Intel NCS2
│   └── scene_analysis.py               # Orchestration layer
├── models/                             # Model storage
│   ├── coral/                          # Coral TFLite models
│   └── intel/                          # OpenVINO IR models
├── venv/                               # Python 3.9 virtual environment
│   └── lib/python3.9/site-packages/openvino/libs/
│       └── libopenvino_intel_myriad_plugin.so  # Custom-built MYRIAD plugin
├── uploads/                            # Temporary file storage
├── tests/                              # Test suite
├── requirements.txt                    # Python dependencies
├── start_server.sh                     # Startup script
├── vision-tool-server.service          # Systemd service definition
├── download_models.py                  # Model downloader
├── install_dependencies.sh             # System setup script
└── test_image_optimization.py          # Image optimization testing
```

---

## Installation Journey

### Critical Discovery: Python 3.9 Requirement

**Problem:** The server failed to start with:
```
ModuleNotFoundError: No module named 'pycoral.pybind._pywrap_coral'
```

**Root Cause:**
- PyCoral requires native C++ extensions (`_pywrap_coral.so`)
- Pre-built wheels only available for Python 3.6-3.9
- System had Python 3.10, incompatible with available wheels
- Building from source required Bazel, which had DNS/network issues

**Solution:**
1. Install Python 3.9 from deadsnakes PPA
2. Recreate virtual environment with Python 3.9
3. Install pre-built pycoral wheels (v2.0.0 for cp39)
4. Reinstall all dependencies in Python 3.9 environment

### Correct Installation Sequence

```bash
# 1. Add deadsnakes PPA for Python 3.9
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.9 python3.9-venv python3.9-dev

# 2. Install system dependencies
sudo apt-get install -y \
    libusb-1.0-0 \
    tesseract-ocr \
    libedgetpu1-std \
    python3-pip

# 3. Create Python 3.9 virtual environment
cd /home/mark/vision-tool-server
python3.9 -m venv venv
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Download and install pycoral wheels
cd /tmp
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
pip install pycoral-2.0.0-cp39-cp39-linux_x86_64.whl tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl

# 7. Download models
cd /home/mark/vision-tool-server
python download_models.py

# 8. Start server
./start_server.sh
```

### Systemd Service Setup

```bash
# The service file is already configured
sudo cp vision-tool-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vision-tool-server
sudo systemctl start vision-tool-server
```

Service configuration highlights:
- **User:** mark
- **WorkingDirectory:** /home/mark/vision-tool-server
- **Python:** /home/mark/vision-tool-server/venv/bin/python
- **Auto-restart:** Always with 10s delay
- **Port:** 8000

---

## Troubleshooting Guide

### Common Issues

#### 1. Server Won't Start - Import Error

**Symptom:**
```
ModuleNotFoundError: No module named 'pycoral.pybind._pywrap_coral'
```

**Cause:** Wrong Python version or missing pycoral native extensions

**Solution:**
```bash
# Check Python version
source venv/bin/activate
python --version  # Must be 3.9.x

# If not 3.9, rebuild venv
deactivate
mv venv venv.backup
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Reinstall pycoral wheels
pip install /tmp/pycoral-2.0.0-cp39-cp39-linux_x86_64.whl \
           /tmp/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
```

#### 2. DNS Resolution Issues During Setup

**Symptom:**
```
curl: (6) Could not resolve host: github.com
```

**Cause:** systemd-resolved issues (intermittent)

**Solution:**
```bash
# Restart DNS resolver
sudo systemctl restart systemd-resolved

# Test resolution
ping -c 2 github.com

# If persistent, use IP directly or wait for network stability
```

#### 3. Docker Container Can't Connect

**Symptom:** OpenWebUI shows "Failed to connect to tool server"

**Cause:** Using wrong IP address (host IP instead of Docker bridge)

**Solution:**
```bash
# From inside OpenWebUI container, server is at Docker bridge gateway
# Correct URL: http://172.17.0.1:8000/openapi.json
# NOT: http://10.0.1.23:8000/openapi.json

# Verify connectivity from container
docker exec open-webui-cuda curl -s http://172.17.0.1:8000/health
```

#### 4. Coral Device Not Detected

**Symptom:** Error when calling object detection/classification

**Check:**
```bash
# List USB devices
lsusb | grep "1a6e"  # Should show Global Unichip Corp

# Check libedgetpu
dpkg -L libedgetpu1-std | grep ".so"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Replug Coral USB
```

#### 5. NCS2 Device Not Detected

**Check:**
```bash
# List USB devices
lsusb | grep "03e7"  # Should show Intel Movidius

# Check OpenVINO installation
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

#### 6. Building OpenVINO MYRIAD Plugin from Source

**Why This Is Needed:**
- Modern OpenVINO (2023+) dropped support for Intel NCS2 (MYRIAD plugin)
- Last version with MYRIAD support: OpenVINO 2022.1.0
- Pre-built 2022.1.0 packages have GCC 11+ compatibility issues
- Solution: Build MYRIAD plugin from source using GCC 9

**Build Process:**

```bash
# 1. Install GCC 9 and build dependencies
sudo apt-get install -y gcc-9 g++-9 cmake git libusb-1.0-0-dev

# 2. Clone OpenVINO 2022.1.0 source
cd /tmp
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout 2022.1.0
git submodule update --init --recursive

# 3. Fix bfloat16 compilation error (GCC 9 uninitialized variable warning)
# Edit src/core/src/type/bfloat16.cpp:60
# Change:
#   uint32_t tmp = (static_cast<uint32_t>(m_value) << 16);
#   float* f = reinterpret_cast<float*>(&tmp);
# To:
#   uint32_t tmp = 0;
#   tmp = (static_cast<uint32_t>(m_value) << 16);
#   float* f = reinterpret_cast<float*>(&tmp);

# 4. Configure build with GCC 9
mkdir build && cd build
CC=gcc-9 CXX=g++-9 cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_INTEL_MYRIAD=ON \
    -DENABLE_INTEL_CPU=OFF \
    -DENABLE_INTEL_GPU=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_SAMPLES=OFF \
    -DENABLE_PYTHON=OFF

# 5. Build MYRIAD plugin only (8 parallel jobs)
make -j8 openvino_intel_myriad_plugin 2>&1 | tee /tmp/myriad_build.log

# 6. Install to system location
sudo make install

# 7. Copy plugin to venv (for isolated Python environment)
cp /opt/intel/openvino_2022.1.0/runtime/lib/intel64/libopenvino_intel_myriad_plugin.so \
   /home/mark/vision-tool-server/venv/lib/python3.9/site-packages/openvino/libs/

# 8. Verify installation
source /home/mark/vision-tool-server/venv/bin/activate
python -c "from openvino.runtime import Core; print(Core().available_devices)"
# Expected output: ['CPU', 'MYRIAD']
```

**Key Build Issues and Fixes:**

1. **Uninitialized Variable Error (bfloat16.cpp:63)**
   - GCC 9 treats warnings as errors with `-Werror`
   - Variable `tmp` used before initialization in type conversion
   - Fix: Initialize `tmp = 0` before assignment

2. **GCC Version Matters**
   - GCC 11+ has stricter warnings about uninitialized variables
   - OpenVINO 2022.1.0 code wasn't written for GCC 11+
   - Use GCC 9 for cleanest build

3. **Plugin Installation Locations**
   - System install: `/opt/intel/openvino_2022.1.0/runtime/lib/intel64/`
   - Python venv: `venv/lib/python3.9/site-packages/openvino/libs/`
   - Both locations may be needed depending on how Python loads OpenVINO

**Verification:**
```bash
# Check if MYRIAD is listed
python -c "from openvino.runtime import Core; core = Core(); print(core.available_devices)"

# Test loading face detection model on MYRIAD
curl -X POST http://localhost:8000/detect_faces -F "file=@test_image.jpg"
```

---

## Maintenance

### Updating Dependencies

```bash
cd /home/mark/vision-tool-server
source venv/bin/activate

# Update specific package
pip install --upgrade package-name

# Freeze current versions
pip freeze > requirements.txt
```

**WARNING:** Do not upgrade pycoral or tflite-runtime without checking compatibility!

### Monitoring

```bash
# View live logs
sudo journalctl -u vision-tool-server -f

# Check service status
sudo systemctl status vision-tool-server

# Recent errors
sudo journalctl -u vision-tool-server --since "1 hour ago" | grep -i error
```

### Performance Tuning

**Current Configuration:**
- uvicorn workers: 1 (single process)
- Max upload size: Default FastAPI (16MB)
- Timeout: None

**To increase workers:**
Edit `start_server.sh`:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

Note: Google Coral can only be accessed by one process at a time!

### Backup Important Files

```bash
# Configuration and code
tar -czf vision-tool-server-backup-$(date +%Y%m%d).tar.gz \
    server.py tools/ *.sh *.service requirements.txt

# Exclude venv and models (can be regenerated)
```

---

## Development Notes

### Adding New Tools

To add a new vision tool:

1. Create module in `tools/new_tool.py`:
```python
def process_image(image_path: str, **params):
    """Process image and return results"""
    # Your logic here
    return {
        "success": True,
        "results": {...}
    }
```

2. Import in `server.py`:
```python
from tools import new_tool
```

3. Add endpoint:
```python
@app.post("/new_endpoint")
async def new_endpoint(file: UploadFile = File(...)):
    # Save file, call tool, return results
    pass
```

4. Update OpenAPI documentation with proper descriptions

### Testing

```bash
# Health check
curl http://localhost:8000/health

# Test object detection
curl -X POST http://localhost:8000/detect_objects \
  -F "file=@test_image.jpg"

# Test with image URL
curl -X POST http://localhost:8000/classify_image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg"}'

# Test image optimization with large image
curl -X POST http://localhost:8000/classify_image \
  -F "file=@large_image.jpg" | jq '.image_metadata.optimization'
```

### Image Token Optimization (November 2, 2025)

**Problem:** Large images can exceed LLM context limits when sent for analysis
- Vision models encode images as tokens
- Typical 4k context limit = ~3500 tokens for images after text overhead
- Large images (4096x3072) = 23,040 tokens = 6x over budget

**Solution:** Automatic geometric scaling with token estimation

**Implementation:**

Location: `/home/mark/vision-tool-server/utils/image_optimizer.py`

**Token Estimation Algorithm:**
```python
def estimate_image_tokens(width: int, height: int) -> int:
    """
    Vision models process images in 512x512 tiles
    Each tile ≈ 400 tokens
    """
    tiles_x = math.ceil(width / 512)
    tiles_y = math.ceil(height / 512)
    total_tiles = tiles_x * tiles_y
    estimated_tokens = total_tiles * 400

    # Add 20% overhead for >4 tiles (positional encoding)
    if total_tiles > 4:
        estimated_tokens = int(estimated_tokens * 1.2)

    return estimated_tokens
```

**Geometric Scaling:**
```python
def calculate_target_dimensions(current_width, current_height, target_tokens):
    """
    Scale by sqrt(target/current) to maintain aspect ratio
    Tokens scale with area, so: new_area/old_area = target/current
    Therefore: scale_factor = sqrt(target/current)
    """
    current_tokens = estimate_image_tokens(current_width, current_height)
    scale_factor = math.sqrt(target_tokens / current_tokens)

    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)

    # Round to multiples of 16 for better model compatibility
    new_width = (new_width // 16) * 16
    new_height = (new_height // 16) * 16

    return new_width, new_height
```

**Exponential Backoff Retry:**
- Attempt 1: Target 3500 tokens (87.5% of 4k context)
- Attempt 2: Target 2800 tokens (70% of 4k context)
- Attempt 3: Target 2000 tokens (50% of 4k context)

**Response Metadata:**
```json
{
  "image_metadata": {
    "original_info": {
      "width": 4096,
      "height": 3072,
      "estimated_tokens": 23040,
      "within_budget": false,
      "recommended_resize": true
    },
    "optimization": {
      "resized": true,
      "original_size": [4096, 3072],
      "new_size": [1584, 1184],
      "original_tokens_estimated": 23040,
      "new_tokens_estimated": 5760,
      "scale_factor": 0.387,
      "token_reduction": "75.0%",
      "attempt": 1,
      "target_tokens": 3500
    }
  }
}
```

**Key Fixes (November 2, 2025):**
1. **Fixed RecursionError:** Circular reference in retry_history metadata
   - Problem: `all_metadata.append(metadata)` then `metadata["retry_history"] = all_metadata`
   - Solution: Create shallow copy without retry_history before appending to history
   - Location: `/home/mark/vision-tool-server/utils/image_optimizer.py:206-226`

2. **JSON Serialization:** Ensured all metadata values are JSON-serializable
   - Changed tuples to lists: `(width, height)` → `[width, height]`
   - Added explicit type conversions: `int()`, `float()`, `str()`
   - Converted Path objects to strings

**Testing:**
```bash
# Create large test image
python -c "
from PIL import Image
img = Image.new('RGB', (4096, 3072), color='white')
img.save('/tmp/test_large.jpg')
"

# Test optimization
curl -X POST http://localhost:8000/classify_image \
  -F "file=@/tmp/test_large.jpg" | jq '.image_metadata'
```

### Environment Variables

Current configuration (hardcoded):
- `HOST`: 0.0.0.0
- `PORT`: 8000
- `UPLOAD_DIR`: ./uploads

To make configurable, add to systemd service:
```ini
[Service]
Environment="VISION_PORT=8000"
Environment="VISION_HOST=0.0.0.0"
```

---

## Integration with OpenWebUI

### Two Integration Methods

**Method 1: Python Tool (Recommended ✅)**
- Install `openwebui_vision_tool.py` as a native Python tool
- Receives image file paths directly from OpenWebUI
- Works with all models that support function calling
- See `OPENWEBUI_TOOL_INSTALLATION.md` for complete guide

**Method 2: External OpenAPI Server (Limited)**
- Register vision server as external tool via `http://10.0.1.23:8000/openapi.json`
- **Limitation:** OpenWebUI doesn't pass images to external OpenAPI tools (as of Nov 2025)
- Useful for direct API access, but not for chat-based image analysis

### Recommended Setup: Python Tool

**Installation:**
1. Copy contents of `openwebui_vision_tool.py`
2. In OpenWebUI: Admin Panel → Tools → Create New Tool
3. Paste code and save
4. Enable tools in chat settings

**Architecture:**
```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  OpenWebUI      │      │  Python Tool     │      │  Vision Server  │
│  (Chat)         │─────►│  (file_handler)  │─────►│  :8000          │
│  Uploads Image  │      │  Reads & Base64  │      │  Coral + NCS2   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                        │                          │
    File saved              File path                  Base64 image
    to disk                 passed                     sent via JSON
```

### Docker Network Architecture

```
┌─────────────────┐         ┌──────────────────┐
│  OpenWebUI      │         │  Vision Server   │
│  Container      │◄───────►│  (Host)          │
│  172.17.0.x     │  Bridge │  10.0.1.23:8000  │
└─────────────────┘         └──────────────────┘
         │                           │
         │                           │
         └───────────┬───────────────┘
                     │
            ┌────────▼─────────┐
            │  Docker Bridge   │
            │  docker0         │
            │  172.17.0.1/16   │
            └──────────────────┘
```

**Key Points:**
- OpenWebUI runs in Docker container on bridge network
- Vision server runs on host (not containerized - needs USB access)
- Python tool calls server at `http://10.0.1.23:8000` (LAN IP)
- CORS enabled for cross-origin requests

### Usage Examples

**Example 1: Object Detection**

**User:** [uploads image of dog]
**User:** "What objects are in this image?"

**Behind the scenes:**
1. OpenWebUI saves image to disk
2. Python tool receives file path via `__files__` parameter
3. Tool reads image, converts to base64
4. Calls `http://10.0.1.23:8000/detect_objects` with base64 data
5. Vision server processes with Google Coral TPU
6. Results returned to tool → formatted → sent to LLM
7. LLM incorporates results in natural language response

**Example 2: Scene Analysis**

**User:** [uploads screenshot with text]
**User:** "Analyze this image completely"

**Behind the scenes:**
1. LLM calls `analyze_scene` tool
2. Tool processes image through all vision APIs:
   - Object detection (Coral)
   - Classification (Coral)
   - OCR text extraction (CPU)
   - Face detection (NCS2)
3. Combined results returned to LLM
4. LLM synthesizes comprehensive description

---

## Security Considerations

### Current State
- ⚠️ **No authentication** - Anyone on network can access
- ⚠️ **No rate limiting** - Potential DoS vector
- ⚠️ **File uploads** - Stored in `./uploads` (not cleaned automatically)
- ✅ **Local processing** - No data leaves the machine
- ✅ **Systemd service** - Runs as user `mark`, not root

### Recommendations for Production

```python
# Add to server.py

# 1. API Key authentication
from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/detect_objects")
async def detect_objects(
    file: UploadFile,
    api_key: str = Depends(api_key_header)
):
    if api_key != os.getenv("VISION_API_KEY"):
        raise HTTPException(401, "Invalid API key")
    # ... rest of code

# 2. Rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/detect_objects")
@limiter.limit("10/minute")
async def detect_objects(...):
    # ... code

# 3. File cleanup
import atexit
import shutil

@atexit.register
def cleanup_uploads():
    shutil.rmtree("uploads", ignore_errors=True)
    os.makedirs("uploads", exist_ok=True)
```

---

## Performance Metrics

### Inference Times (Approximate)

| Operation | Hardware | Time | Notes |
|-----------|----------|------|-------|
| Object Detection | Coral | 15-30ms | SSD MobileNet v2 |
| Classification | Coral | 5-10ms | MobileNet v2 |
| Face Detection | NCS2 | 40-60ms | retail-0004 model |
| OCR (EasyOCR) | CPU | 1-3s | Depends on text density |
| Scene Analysis | All | 2-4s | Combined pipeline |

### Resource Usage

- **Memory:** ~400MB resident (with models loaded)
- **CPU:** Minimal (inference on accelerators)
- **Storage:** ~500MB (models + venv)

---

## Known Issues

1. **Python 3.10+ incompatibility** - pycoral wheels not available
2. **Single Coral access** - Only one process can use Coral at a time
3. **No model caching** - Models loaded on first request (small delay)
4. **Upload directory growth** - Files not auto-deleted (implement cleanup)

---

## Future Enhancements

- [ ] Add authentication/API keys
- [ ] Implement rate limiting
- [ ] Add model caching/warm-up
- [ ] Support batch processing
- [ ] Add video frame analysis
- [ ] Implement webhook callbacks
- [ ] Add prometheus metrics
- [ ] Create Docker container (with USB passthrough)
- [ ] Add queue system for concurrent requests
- [ ] Implement result caching

---

## References

- [Google Coral Documentation](https://coral.ai/docs/)
- [pycoral GitHub](https://github.com/google-coral/pycoral)
- [Intel OpenVINO](https://docs.openvino.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenWebUI Tools](https://docs.openwebui.com/)

---

**Last Updated:** November 3, 2025
**Maintained By:** Claude (AI Assistant)
**Contact:** User `mark` on system `ml-compute02`

---

## Changelog

### November 3, 2025
- ✅ Created OpenWebUI Python tool wrapper (`openwebui_vision_tool.py`)
- ✅ Added CORS support to FastAPI server
- ✅ Converted all endpoints to support JSON with base64 images
- ✅ Added comprehensive installation guide (`OPENWEBUI_TOOL_INSTALLATION.md`)
- ✅ Discovered limitation: External OpenAPI tools don't receive images from OpenWebUI
- ✅ Solution: Use Python tool with `file_handler = True` instead

### November 2, 2025
- ✅ Implemented image token optimization system
- ✅ Built custom OpenVINO 2022.1.0 MYRIAD plugin for NCS2
- ✅ Added automatic image resizing with exponential backoff
- ✅ Fixed RecursionError in retry metadata

### November 1, 2025
- ✅ Initial project setup
- ✅ Installed Python 3.9 environment
- ✅ Configured Google Coral TPU support
- ✅ Configured Intel NCS2 support
- ✅ Created FastAPI server with all vision endpoints
- ✅ Set up systemd service
- ✅ Downloaded and configured vision models
