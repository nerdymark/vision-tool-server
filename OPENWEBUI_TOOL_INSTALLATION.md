# Installing Vision Tools in OpenWebUI

This guide shows you how to install the local vision analysis tools (Google Coral + Intel NCS2) as a Python tool in OpenWebUI.

## Prerequisites

1. ✅ Vision Tool Server running at `http://10.0.1.23:8000`
2. ✅ OpenWebUI installed and accessible
3. ✅ Admin access to OpenWebUI (for installing tools)

## Installation Steps

### Step 1: Copy the Tool Code

1. Open the file `/home/mark/vision-tool-server/openwebui_vision_tool.py`
2. Copy the entire contents

### Step 2: Install in OpenWebUI

1. **Navigate to Tools**
   - In OpenWebUI, click on your profile icon (top right)
   - Go to **Admin Panel** → **Tools**
   - Or go to **Workspace** → **Tools**

2. **Create New Tool**
   - Click the **"+"** button or **"Create New Tool"**

3. **Paste the Code**
   - Paste the entire contents of `openwebui_vision_tool.py` into the editor

4. **Save the Tool**
   - Click **"Save"** or **"Create"**

### Step 3: Configure (Optional)

The tool has configurable settings (Valves):

- **VISION_SERVER_URL**: Default is `http://10.0.1.23:8000`
  - Change this if your vision server is at a different address
  - Use `http://localhost:8000` if OpenWebUI and vision server are on same machine
  - Use `http://172.17.0.1:8000` if accessing from inside Docker

- **OBJECT_DETECTION_THRESHOLD**: Default is `0.4` (40% confidence)
  - Higher = fewer but more accurate detections
  - Lower = more detections but less confident

- **FACE_DETECTION_THRESHOLD**: Default is `0.5` (50% confidence)

To configure:
1. Go to **Tools** in OpenWebUI
2. Find "Vision Tool" and click the settings/gear icon
3. Adjust the values under "Valves"

### Step 4: Enable in Chat

1. **Start a new chat** or open an existing one

2. **Enable the tools**
   - Look for the **"+"** icon or **"Tools"** button
   - Select which vision tools to enable:
     - `detect_objects` - Find objects in images
     - `classify_image` - Identify what the image contains
     - `extract_text` - OCR text extraction
     - `detect_faces` - Detect faces
     - `analyze_scene` - Comprehensive analysis (all of the above)

3. **Enable Native Function Calling** (Recommended)
   - Click the settings/parameters icon in the chat
   - Go to **Advanced Params**
   - Change **Function Calling** from "Default" to "Native"

### Step 5: Use a Compatible Model

For best results, use a model that supports function calling:

**Ollama models (local):**
- `llama3.1:latest` ✅
- `llama3.2:latest` ✅
- `llama3.2-vision:latest` ✅ (has built-in vision too)
- `mistral-nemo:latest` ✅
- `qwen2.5:latest` ✅

**API models:**
- GPT-4o, GPT-4-turbo ✅
- Claude 3.5 Sonnet ✅
- Gemini Pro ✅

## Usage Examples

### Example 1: Detect Objects

1. Upload an image (click the "+" icon → Upload Files)
2. Ask: "What objects are in this image?"
3. The LLM will automatically call `detect_objects` tool
4. Results will show detected objects with confidence scores

### Example 2: Read Text from Image

1. Upload an image with text (screenshot, photo of document, etc.)
2. Ask: "What text is in this image?"
3. The LLM will call `extract_text` tool
4. Extracted text will be displayed

### Example 3: Comprehensive Analysis

1. Upload any image
2. Ask: "Analyze this image completely"
3. The LLM will call `analyze_scene` tool
4. You'll get objects, classification, text, and faces detected

### Example 4: Manual Tool Call

You can also explicitly request a tool:

```
Use the detect_faces tool to find faces in the uploaded image
```

## Available Tools

| Tool | Description | Hardware Used |
|------|-------------|---------------|
| `detect_objects` | Detect objects with bounding boxes | Google Coral TPU |
| `classify_image` | Classify image into 1000+ categories | Google Coral TPU |
| `extract_text` | OCR text extraction (multi-language) | CPU (EasyOCR) |
| `detect_faces` | Detect faces with locations | Intel NCS2 |
| `analyze_scene` | Complete analysis (all of above) | All hardware |

## Troubleshooting

### Tool Not Appearing

- Make sure you saved the tool after pasting the code
- Refresh the OpenWebUI page
- Check that you're in a chat where tools are enabled

### "Error: Vision API error: Connection refused"

- Verify the vision server is running: `sudo systemctl status vision-tool-server`
- Check the server URL in tool settings (Valves)
- Test connectivity: `curl http://10.0.1.23:8000/health`

### "Error: No image provided"

- Make sure you uploaded an image before asking the question
- The image must be visible in the chat
- Try re-uploading the image

### Model Not Calling Tools

- Enable **Native Function Calling** in Advanced Params
- Use a model that supports function calling (see list above)
- Try being more explicit: "Use the detect_objects tool on the image"

### Tool Returns Empty Results

- The image might not contain what you're looking for
- Try adjusting confidence thresholds in tool settings (lower = more detections)
- Check image quality (very small or blurry images may not detect well)

## Advanced: Custom Configuration

### Change Server URL for Docker

If OpenWebUI is running in Docker:

1. Edit the tool in OpenWebUI
2. Change `VISION_SERVER_URL` valve to:
   - `http://172.17.0.1:8000` (Docker bridge gateway)
   - Or use the host machine's LAN IP: `http://10.0.1.23:8000`

### Adjust Detection Sensitivity

For more sensitive object detection:
1. Go to tool settings
2. Set `OBJECT_DETECTION_THRESHOLD` to `0.3` (from default `0.4`)

For fewer false positives:
1. Set threshold to `0.6` or higher

## Technical Details

### How It Works

1. You upload an image to OpenWebUI
2. OpenWebUI saves it to disk and passes the file path to the tool
3. The Python tool reads the image and converts it to base64
4. It sends the base64 data to the vision server API at `http://10.0.1.23:8000`
5. The vision server processes it using:
   - Google Coral TPU for object detection & classification
   - Intel NCS2 for face detection
   - EasyOCR (CPU) for text extraction
6. Results are returned to the tool
7. The tool formats results as human-readable text
8. OpenWebUI displays the results in the chat

### Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│  OpenWebUI  │─────▶│ Python Tool  │─────▶│  Vision Server  │
│   (Chat)    │      │  (This File) │      │  :8000          │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │                        │
                     File Path                  Base64 Image
                                                     │
                                          ┌──────────┴─────────┐
                                          │                    │
                                    Google Coral          Intel NCS2
                                    (Objects/Class)       (Faces)
```

## Performance

- **Object Detection**: ~15-30ms per image
- **Classification**: ~5-10ms per image
- **Face Detection**: ~40-60ms per image
- **OCR**: ~1-3 seconds (depends on text amount)
- **Scene Analysis**: ~2-4 seconds (combined)

All processing is local - no cloud services or external APIs!

## Support

If you encounter issues:

1. Check vision server logs: `sudo journalctl -u vision-tool-server -f`
2. Verify server health: `curl http://10.0.1.23:8000/health`
3. Test direct API:
   ```bash
   curl -X POST http://10.0.1.23:8000/detect_objects \
     -H "Content-Type: application/json" \
     -d "{\"image_base64\": \"$(base64 -w 0 test.jpg)\"}"
   ```

## Privacy & Security

✅ **All processing is local** - images never leave your network
✅ **No cloud services** - completely offline capable
✅ **No API keys needed** - uses local hardware accelerators
✅ **Fast inference** - hardware acceleration via Coral & NCS2

---

**Created**: November 3, 2025
**Version**: 1.0
**Hardware**: Google Coral USB + Intel NCS2
