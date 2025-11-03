#!/bin/bash
# OpenWebUI Vision Tool Server Diagnostic Script
# Checks connectivity, configuration, and tool registration

set -e

echo "==============================================="
echo "OpenWebUI Vision Tool Server Diagnostics"
echo "==============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Vision Tool Server Status
echo "1. Checking Vision Tool Server..."
if systemctl is-active --quiet vision-tool-server; then
    echo -e "${GREEN}✓${NC} Service is running"
else
    echo -e "${RED}✗${NC} Service is NOT running"
    echo "   Run: sudo systemctl start vision-tool-server"
    exit 1
fi

# Check 2: Health Endpoint
echo ""
echo "2. Checking Health Endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health || echo "FAILED")
if [[ "$HEALTH_RESPONSE" == "FAILED" ]]; then
    echo -e "${RED}✗${NC} Cannot reach http://localhost:8000/health"
    exit 1
else
    echo -e "${GREEN}✓${NC} Health endpoint responding"
    echo "   Available tools:"
    echo "$HEALTH_RESPONSE" | python3 -m json.tool | grep '"available": true' -B1 | grep -E '(object_detection|classification|ocr|face_detection)' || true
fi

# Check 3: OpenAPI Spec
echo ""
echo "3. Checking OpenAPI Specification..."
OPENAPI_RESPONSE=$(curl -s http://localhost:8000/openapi.json || echo "FAILED")
if [[ "$OPENAPI_RESPONSE" == "FAILED" ]]; then
    echo -e "${RED}✗${NC} Cannot fetch OpenAPI spec"
    exit 1
else
    TOOL_COUNT=$(echo "$OPENAPI_RESPONSE" | grep -o '"operationId"' | wc -l)
    echo -e "${GREEN}✓${NC} OpenAPI spec available ($TOOL_COUNT endpoints)"
fi

# Check 4: OpenWebUI Container
echo ""
echo "4. Checking OpenWebUI Container..."
if docker ps | grep -q open-webui; then
    echo -e "${GREEN}✓${NC} OpenWebUI container is running"
    CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep open-webui | head -1)
    echo "   Container: $CONTAINER_NAME"
else
    echo -e "${RED}✗${NC} OpenWebUI container is NOT running"
    echo "   Cannot test container connectivity"
fi

# Check 5: Container Connectivity (if OpenWebUI is running)
if docker ps | grep -q open-webui; then
    echo ""
    echo "5. Checking Container → Tool Server Connectivity..."
    CONTAINER_NAME=$(docker ps --format '{{.Names}}' | grep open-webui | head -1)
    CONTAINER_HEALTH=$(docker exec "$CONTAINER_NAME" curl -s http://172.17.0.1:8000/health 2>&1 || echo "FAILED")

    if [[ "$CONTAINER_HEALTH" == *"FAILED"* ]] || [[ "$CONTAINER_HEALTH" == *"curl"* ]]; then
        echo -e "${RED}✗${NC} Container cannot reach http://172.17.0.1:8000/health"
        echo "   This is the most common issue!"
        echo "   Solution: Ensure Docker bridge IP is correct (usually 172.17.0.1)"
    else
        echo -e "${GREEN}✓${NC} Container can reach tool server via Docker bridge"
    fi
fi

# Check 6: Tool Registration in OpenWebUI
echo ""
echo "6. Checking Tool Registration..."
echo -e "${YELLOW}ℹ${NC}  Manual step required:"
echo "   1. Open OpenWebUI: http://localhost:3000"
echo "   2. Go to: Admin Panel → Settings → Tools"
echo "   3. Add tool server URL: http://172.17.0.1:8000/openapi.json"
echo "   4. Click 'Refresh' or restart OpenWebUI container"
echo ""
echo "   To restart OpenWebUI:"
echo "   docker restart open-webui-cuda"

# Check 7: Available Ollama Models
echo ""
echo "7. Checking Ollama Models..."
if command -v ollama &> /dev/null; then
    echo "   Available models with tool calling support:"
    ollama list | grep -E 'llama3|llava|minicpm|qwen' || echo "   No vision/tool-calling models found"
    echo ""
    echo -e "${YELLOW}ℹ${NC}  Recommended models for vision tools:"
    echo "   - llama3.2-vision (best for tool calling)"
    echo "   - llava (good for vision)"
    echo "   - qwen2-vl (excellent vision model)"
    echo "   - minicpm-v (lightweight vision model)"
else
    echo -e "${YELLOW}⚠${NC}  Ollama not found in PATH"
fi

# Check 8: Hardware Status
echo ""
echo "8. Checking Hardware Accelerators..."
# Coral TPU
if lsusb | grep -q "18d1:9302"; then
    echo -e "${GREEN}✓${NC} Google Coral USB Accelerator detected"
else
    echo -e "${YELLOW}⚠${NC}  Google Coral USB Accelerator not detected"
fi

# Intel NCS2
if lsusb | grep -q "Movidius MyriadX"; then
    echo -e "${GREEN}✓${NC} Intel NCS2 (Movidius MyriadX) detected"
    # Check permissions
    NCS2_DEV=$(lsusb | grep "Movidius MyriadX" | sed 's/Bus \([0-9]*\) Device \([0-9]*\).*/\/dev\/bus\/usb\/\1\/\2/')
    if [ -r "$NCS2_DEV" ]; then
        echo "   Permissions: OK"
    else
        echo -e "${YELLOW}⚠${NC}   Permission issue - user may not be in 'users' group"
        echo "   Run: sudo usermod -aG users $USER"
        echo "   Then logout and login again"
    fi
else
    echo -e "${YELLOW}⚠${NC}  Intel NCS2 not detected"
fi

# Summary
echo ""
echo "==============================================="
echo "Diagnostic Summary"
echo "==============================================="
echo ""
echo "If all checks passed:"
echo "1. Ensure tools are registered in OpenWebUI (see step 6 above)"
echo "2. In OpenWebUI chat, click the plugin icon (🔧) to enable tools"
echo "3. Upload an image and ask about it"
echo ""
echo "Common issues:"
echo "- Tools not showing: Restart OpenWebUI container after adding tool URL"
echo "- Models not using tools: Enable tools via plugin icon in chat"
echo "- Connection errors: Check Docker bridge IP (172.17.0.1)"
echo ""
echo "For more help, check:"
echo "- Server logs: sudo journalctl -u vision-tool-server -f"
echo "- API docs: http://localhost:8000/docs"
echo "- GitHub: https://github.com/nerdymark/vision-tool-server"
echo ""
