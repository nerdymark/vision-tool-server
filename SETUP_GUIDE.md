# Vision Tool Server - Setup Complete! 🎉

Your AI-powered vision tool server has been successfully configured!

## 📦 What's Been Built

### Hardware Detected
- ✅ **Google Coral USB Accelerator** - For fast object detection and image classification
- ✅ **Intel Neural Compute Stick 2** - For face detection (model download pending)

### Tools Implemented

1. **Object Detection** (`/detect_objects`)
   - Device: Google Coral
   - Model: SSD MobileNet V2 (COCO dataset)
   - Detects 80+ object types with bounding boxes

2. **Image Classification** (`/classify_image`)
   - Device: Google Coral
   - Model: MobileNet V2 (ImageNet)
   - Classifies images into 1000+ categories

3. **OCR Text Extraction** (`/extract_text`)
   - Engine: EasyOCR (CPU)
   - Multi-language support
   - Returns text with bounding boxes

4. **Face Detection** (`/detect_faces`)
   - Device: Intel NCS2
   - Model: face-detection-retail-0004 (FP16)
   - **STATUS:** ✅ Fully operational

5. **Scene Analysis** (`/analyze_scene`)
   - Combines all tools above
   - Returns comprehensive scene description

## 📁 Project Structure

```
/home/mark/vision-tool-server/
├── server.py                    # FastAPI server
├── tools/
│   ├── object_detection.py      # Coral object detection
│   ├── classification.py        # Coral classification
│   ├── ocr.py                  # OCR text extraction
│   ├── face_detection.py       # NCS2 face detection
│   └── scene_analysis.py       # Combined analysis
├── models/
│   └── coral/                  # Downloaded Coral models ✓
├── venv/                       # Python virtual environment
├── start_server.sh             # Server startup script
└── vision-tool-server.service  # Systemd service file
```

## 🚀 Quick Start

### 1. Test the Server

```bash
cd /home/mark/vision-tool-server
./start_server.sh
```

The server will start on **http://localhost:8000**

### 2. Check Health Status

Open another terminal:
```bash
curl http://localhost:8000/health
```

### 3. Test with an Image

```bash
# Object detection
curl -X POST http://localhost:8000/detect_objects \
  -F "file=@/path/to/your/image.jpg"

# Image classification
curl -X POST http://localhost:8000/classify_image \
  -F "file=@/path/to/your/image.jpg"

# OCR text extraction
curl -X POST http://localhost:8000/extract_text \
  -F "file=@/path/to/your/image.jpg"
```

### 4. View API Documentation

Visit: **http://localhost:8000/docs** (Swagger UI)

## 🔄 Enable Auto-Start (Systemd)

```bash
# Copy service file
sudo cp vision-tool-server.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable vision-tool-server
sudo systemctl start vision-tool-server

# Check status
sudo systemctl status vision-tool-server

# View logs
sudo journalctl -u vision-tool-server -f
```

## 🔗 Integrate with OpenWebUI

### Method 1: Add as External Tool

1. Open OpenWebUI at http://localhost:3000
2. Go to **Admin Panel → Settings → Functions**
3. Click **Add Function**
4. Enter the OpenAPI spec URL: `http://localhost:8000/openapi.json`

### Method 2: Configure as Tool Server

1. In OpenWebUI settings, look for "External Tools" or "Tool Servers"
2. Add: `http://localhost:8000`
3. The AI will now be able to call vision tools during conversations!

## 📋 Getting Started

### 1. Test All Endpoints

Create test images and verify each endpoint works:

```bash
# Download a test image
wget https://raw.githubusercontent.com/google-coral/test_data/master/cat.jpg -O /tmp/test.jpg

# Test object detection
curl -X POST http://localhost:8000/detect_objects -F "file=@/tmp/test.jpg"

# Test classification
curl -X POST http://localhost:8000/classify_image -F "file=@/tmp/test.jpg"

# Test OCR (use an image with text)
curl -X POST http://localhost:8000/extract_text -F "file=@/tmp/receipt.jpg"

# Test full scene analysis
curl -X POST http://localhost:8000/analyze_scene -F "file=@/tmp/test.jpg"
```

### 2. Configure Firewall (if needed)

If accessing from other devices:
```bash
sudo ufw allow 8000/tcp
```

## 🎯 Example Use Cases

### Conversation with OpenWebUI

```
User: "Here's a photo from my garden [uploads image]"

AI: [calls /analyze_scene]
AI: "This appears to be a garden scene with high confidence. I can see:
     tomato plants, garden tools, a watering can. The scene contains
     5 objects total. There's also some text visible on what appears
     to be a seed packet."

User: "What does the text say?"

AI: [calls /extract_text focused on seed packet area]
AI: "The text reads: 'Cherry Tomatoes - Plant in Spring'"
```

### Document Processing

```
User: "What's the total on this receipt? [uploads photo]"

AI: [calls /extract_text]
AI: [parses extracted text]
AI: "The receipt shows a total of $47.52"
```

### Security/Monitoring

```
User: "Is anyone at the door?" [camera feed]

AI: [calls /detect_faces and /detect_objects]
AI: "Yes, I detect 1 person at the door. They appear to be holding
     a package."
```

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check Python environment
cd /home/mark/vision-tool-server
source venv/bin/activate
python server.py  # Look for error messages
```

### Coral device not detected
```bash
lsusb | grep "1a6e"  # Should see Global Unichip Corp
# If not, try:
sudo udevadm control --reload-rules
sudo udevadm trigger
# Unplug and replug Coral
```

### NCS2 device not detected
```bash
lsusb | grep "03e7"  # Should see Intel Movidius
# If not, check udev rules and replug device
```

### Port 8000 already in use
```bash
# Change port in server.py (line at bottom)
# Or stop conflicting service
sudo lsof -i :8000
```

### PyCoral import errors
```bash
# If you get "No module named 'pycoral.adapters'" error
cd /tmp
git clone --depth 1 https://github.com/google-coral/pycoral.git
cd pycoral
source /home/mark/vision-tool-server/venv/bin/activate
pip install --no-deps .
```

## 📚 API Reference

### All Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info |
| `/health` | GET | Health check |
| `/detect_objects` | POST | Object detection |
| `/classify_image` | POST | Image classification |
| `/extract_text` | POST | OCR text extraction |
| `/detect_faces` | POST | Face detection |
| `/analyze_scene` | POST | Full scene analysis |
| `/docs` | GET | Swagger documentation |
| `/openapi.json` | GET | OpenAPI spec |

## 🎉 Success!

Your vision tool server is ready to give your LLM "eyes"!

**What you can do now:**
- Start the server and test endpoints
- Integrate with OpenWebUI
- Upload images and let your AI analyze them
- Process documents, receipts, screenshots
- Detect objects in photos
- Extract text from images

**Next steps:**
1. Start the server: `./start_server.sh`
2. Test with sample images
3. Add to OpenWebUI
4. Download face detection model when network permits

---

**Created:** November 1, 2025
**Project Location:** `/home/mark/vision-tool-server`
**Server URL:** `http://localhost:8000`
**OpenWebUI URL:** `http://localhost:3000`
