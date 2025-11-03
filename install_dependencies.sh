#!/bin/bash
# Installation script for Vision Tool Server dependencies

set -e

echo "=== Installing System Dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    libusb-1.0-0 libusb-1.0-0-dev \
    udev \
    tesseract-ocr \
    cmake build-essential \
    wget curl gnupg \
    libopencv-dev \
    pkg-config

echo ""
echo "=== Installing Google Coral Edge TPU Runtime ==="
# Add Coral repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

# Install libedgetpu (standard version, not max frequency to avoid overheating)
sudo apt-get install -y libedgetpu1-std

# Set up udev rules for Coral
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", ATTRS{idProduct}=="089a", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-coral.rules
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-coral.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Add current user to plugdev group
sudo usermod -a -G plugdev $USER

echo ""
echo "=== Installing Intel OpenVINO Dependencies ==="
# OpenVINO will be installed via pip, but we need some system libraries
sudo apt-get install -y \
    libgtk-3-0 \
    libtbb2 \
    libpugixml1v5

# Set up udev rules for Intel NCS2
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/97-myriad-usbboot.rules
sudo udevadm control --reload-rules
sudo udevadm trigger

echo ""
echo "=== System dependencies installed successfully ==="
echo ""
echo "IMPORTANT: You may need to log out and log back in for group changes to take effect."
echo "           Or run: newgrp plugdev"
echo ""
